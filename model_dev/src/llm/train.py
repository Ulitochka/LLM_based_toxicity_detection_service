import argparse
import os
import warnings
import time
import random
from uuid import uuid4
import csv
import gc

import numpy as np
from datasets import Dataset
from omegaconf import ListConfig
from hydra.experimental import compose, initialize
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments)
import torch.optim as optim
import bitsandbytes as bnb
import bitsandbytes.optim as bnb_optim
from transformers.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, replace_lora_weights_loftq
from peft.tuners.lora.config import CordaConfig
from peft.tuners.lora.corda import preprocess_corda
from sklearn.metrics import f1_score, precision_score, recall_score

from data_tools.cls_preprocessor import ClsPreprocessor
from metrics.tag_soft_metrics import TagSoftMetrics
from metrics.cls_metrics import ClsMetrics
from data_tools.dataset import Dataset
from data_tools.data_collator_chat import DataCollatorChat
from data_tools.data_collator import DataCollator


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
disable_progress_bar()
random.seed(42)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Trainer.tokenizer is now deprecated.*")


def read_text_file(path):
    with open(path) as f:
        return [el.strip() for el in f.readlines()]
    

def read_csv_as_dicts(path, delimiter=',') -> list:
    data = []
    with open(path) as f:
        lines = csv.DictReader(f, delimiter=delimiter)
        for row in lines:
            data.append(row)
    return data
    

def find_all_linear_names(model: AutoModelForCausalLM) -> list:
    if model_config.quant4:
        cls = bnb.nn.Linear4bit
    if model_config.quant8:
        cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def to_native(value):
    if isinstance(value, ListConfig):
        return list(value)
    return value


def pred_postprocess(preds: list[list]) -> list[str]:
    result = []
    for el in preds:
        text = ''.join(el).replace('\n', '')
        pred = text.split('|||')[-1].strip()
        if pred in label2id:
            pred = label2id[pred]
        else:
            pred = 0
        result.append(pred)
    return result


def run_model():
    for batch in train_sample_dataloader:
        prep_data = {
            "input_ids": batch["input_ids"].to(device), 
            "attention_mask": batch["attention_mask"].to(device)}
        with torch.no_grad():
            model(**prep_data)


def run_evaluation(
        peft_model,
        inference_dataloader,
        tokenizer,
        label2id,
        data_config,
        device):

    peft_model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_inference in tqdm(
            inference_dataloader, total=len(inference_dataloader)):

            prep_data = {
                "input_ids": batch_inference["input_ids_inference"].to(device), 
                "attention_mask": batch_inference["attention_mask_inference"].to(device)}

            preds = peft_model.generate(
                **prep_data,
                max_new_tokens=data_config.max_target_len,
                do_sample=False,
                use_cache=False,
                cache_implementation=None)
            
            preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            preds = pred_postprocess(preds)

            target_texts = np.where(batch_inference["labels"] != -100, batch_inference["labels"], tokenizer.pad_token_id)
            target_texts = tokenizer.batch_decode(target_texts, skip_special_tokens=True)
            target_texts = [label2id[el.strip()] for el in target_texts]

            all_predictions.extend(preds)
            all_labels.extend(target_texts)

            del (prep_data, preds, target_texts, batch_inference)


    metrics = {
        "f1": f1_score(all_labels, all_predictions),
        "precision": precision_score(all_labels, all_predictions),
        "recall": recall_score(all_labels, all_predictions)
    }

    print(f"[Evaluation] F1: {metrics['f1']} | Precision: {metrics['precision']} | Recall: {metrics['recall']}")
    print('-' * 100)
              
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    del (all_labels, all_predictions)

    return metrics



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True)
    parser.add_argument('--data_set_path', type=str, required=True)
    parser.add_argument('--limit', action='store_true', required=False)
    parser.add_argument('--wandb', action='store_true', required=False)
    parser.add_argument('--config_id', type=str, required=True)
    parser.add_argument('--custom_trainer', action='store_true', required=False)
    parser.add_argument('--mixed_precision', action='store_true', required=False)
    args = parser.parse_args()

    initialize(config_path='../../tune_configs')
    train_config = compose(config_name=args.config_id)['train_config']
    optimizer_config = compose(config_name=args.config_id)['optimizer_config']
    model_config = compose(config_name=args.config_id)['model_config']
    lora_config = compose(config_name=args.config_id)['lora_config']
    data_config = compose(config_name=args.config_id)['data_config']

    login(token=model_config.token)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision('high')
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    else:
        device = torch.device("cpu")

    experiment_id = time.strftime(f'%Y_%m_%d-%H_%M_%S') + str(uuid4())[:8] + f"_{model_config.model.replace('/', '_')}"
    new_res_path = os.path.join(args.experiment_path, f"{experiment_id}")
    if not os.path.exists(new_res_path):
        os.makedirs(new_res_path)

    train_args = TrainingArguments(output_dir="./default")
    for el in [('train_config', train_config), ('optimizer_config', optimizer_config)]:
        for key, value in el[1].items():
            if hasattr(train_args, key):
                setattr(train_args, key, value)
                print(f"Установлено: from {el[0]} {key} = {value}")
            else:
                print(f"Пропущено:  from {el[0]} {key} — не найден в TrainingArguments")

    for key, value in model_config.items():
        if key != 'token':
            print(f"Установлено: from model_config {key} = {value}")

    setattr(train_args, 'output_dir', new_res_path)

    if args.wandb:
        import wandb
        os.environ["WANDB_DISABLED"] = "false"
        setattr(train_args, 'report_to', 'wandb')

        wandb.init(
            project=f"{data_config.task_type}",
            name=experiment_id,
            config={
                'train_config': train_config,
                'optimizer_config': optimizer_config,
                'model_config': model_config,
                'data_config': data_config})

    tokenizer = AutoTokenizer.from_pretrained(model_config.model, token=model_config.token)

    print("EOS Token:", tokenizer.eos_token)
    print("EOS Token ID:", tokenizer.eos_token_id)
    print("PAD Token:", tokenizer.pad_token)
    print("PAD Token ID:", tokenizer.pad_token_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Change pad_token:", tokenizer.pad_token_id)

    if 'cls' in data_config.task_type:

        metrics = ClsMetrics(
            tokenizer=tokenizer,
            res_path=new_res_path,
            report_to=train_args.report_to)

        preprocessor = ClsPreprocessor(
            tokenizer,
            data_config.prefix,
            data_config.max_input_len)

        train_data = read_csv_as_dicts(os.path.join(args.data_set_path, 'train.csv'))
        test_data = read_csv_as_dicts(os.path.join(args.data_set_path, 'test.csv'))
        print(test_data[0])

        if args.limit:
            test_data = test_data[:200]
            train_data = train_data[:200]

        print('train_data: ', len(train_data))
        print('test_data: ', len(test_data))
        print('max_input_len: ', data_config.max_input_len)

        if data_config.collator_type == 'chat':
            train_data = [preprocessor.transform_example_to_chat(el) for el in train_data]
            test_data = [preprocessor.transform_example_to_chat(el) for el in test_data]
        else:
            train_data = [preprocessor.transform_example(el) for el in train_data]
            test_data = [preprocessor.transform_example(el) for el in test_data]

        if data_config.collator_type == 'chat':
            data_collator = DataCollatorChat(
                tokenizer=tokenizer)
        else:
            data_collator = DataCollator(
                tokenizer=tokenizer,
                preprocessor=preprocessor,
                max_input_len=data_config.max_input_len,
                max_target_len=data_config.max_target_len)
            
        train_sample = random.sample(train_data, 500)
        train_sample_dataloader = DataLoader(
            train_sample,
            batch_size=train_config.per_device_train_batch_size,
            shuffle=False,
            collate_fn=data_collator.collate_fn)
    
    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    if model_config.quant8:
      bnb_config = BitsAndBytesConfig(
          load_in_8bit=True,
          llm_int8_threshold=6.0,
          llm_int8_skip_modules=["lm_head"])

    if model_config.quant4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        token=model_config.token,
        attn_implementation='eager',
        use_cache=False)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    print('Model [X]')

    target_modules = to_native(lora_config.target_modules)
    if not target_modules:
        target_modules = find_all_linear_names(model)
    print('target_modules:', target_modules)
    print('init_lora_weights:', lora_config.init_lora_weights)
    print('lora_r:', lora_config.lora_r)
    print('lora_alpha:', lora_config.lora_alpha)

    if lora_config.init_lora_weights == 'corda':
        corda_config = CordaConfig(
            corda_method="kpm",
            cache_file=f'{new_res_path}/corda_cache')

    lora_config = LoraConfig(
        task_type=lora_config.task_type,
        r=lora_config.lora_r,
        bias=lora_config.lora_bias,
        lora_alpha=lora_config.lora_alpha,
        target_modules=target_modules,
        init_lora_weights=lora_config.init_lora_weights)
    
    if lora_config.init_lora_weights == 'corda':
        lora_config.corda_config = corda_config
        preprocess_corda(model, lora_config, run_model=run_model)

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    print('Peft model [X]')

    if args.custom_trainer:

        peft_model.gradient_checkpointing_enable()

        scaler = torch.cuda.amp.GradScaler()

        label2id = {
            'да': 1,
            'нет': 0,
        }

        decay_params = []
        no_decay_params = []

        for name, param in peft_model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "layernorm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": optimizer_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},]
        
        if optimizer_config.optim == 'adam_8bit':
            optimizer = bnb_optim.AdamW8bit(
                model.parameters(),
                lr=optimizer_config.learning_rate)
        else:
            optimizer = optim.AdamW(
                optimizer_grouped_parameters,
                lr=optimizer_config.learning_rate)
        
        train_dataset = Dataset(train_data)
        test_dataset = Dataset(test_data)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=data_collator.collate_fn)
        
        inference_dataloader = DataLoader(
            test_dataset,
            batch_size=train_config.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=data_collator.collate_fn_inference)
        
        total_steps = len(train_dataloader)

        f1_best = 0

        for epoch in range(train_config.num_train_epochs):

            peft_model.train()
            
            for step, batch in enumerate(train_dataloader):

                prep_data = {
                    "input_ids": batch["input_ids"].to(device), 
                    "attention_mask": batch["attention_mask"].to(device),
                    "labels": batch["labels"].to(device)}
                
                if args.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = peft_model(**prep_data)
                        loss = outputs.loss / train_config.gradient_accumulation_steps
                        scaler.scale(loss).backward()
                else:
                    outputs = peft_model(**prep_data)
                    loss = outputs.loss / train_config.gradient_accumulation_steps
                    loss.backward()

                if (step + 1) % train_config.logging_steps == 0:
                    print(f"Epoch {epoch} | Step {step + 1}/{total_steps} | Loss: {loss.item():.4f}")

                if (step + 1) % train_config.gradient_accumulation_steps == 0:
                    if args.mixed_precision:
                        # для нормального клипа градиентов
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            peft_model.parameters(),
                            max_norm=optimizer_config.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            peft_model.parameters(),
                            max_norm=optimizer_config.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()
                
                del (prep_data, outputs, batch, loss)

                if (step + 1) % train_config.eval_steps == 0:

                    metrics = run_evaluation(
                        peft_model=peft_model,
                        inference_dataloader=inference_dataloader,
                        tokenizer=tokenizer,
                        label2id=label2id,
                        data_config=data_config,
                        device=device,
                        epoch=epoch,
                        args=args)
                    
                    if metrics['f1'] > f1_best:
                        f1_best = metrics['f1']
                        peft_model.model.save_pretrained(f"{new_res_path}/{epoch}")
                        peft_model.save_pretrained(f"{new_res_path}/{epoch}")
                        tokenizer.save_pretrained(f"{new_res_path}/{epoch}")
                        print('Save new best model!')
                    
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

    else:
        trainer = Trainer(
            model=peft_model,
            args=train_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            processing_class=tokenizer,
            data_collator=data_collator.collate_fn)

        trainer_stats = trainer.train()

    if args.wandb:
        wandb.finish()
    
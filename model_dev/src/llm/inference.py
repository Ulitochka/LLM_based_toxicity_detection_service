import argparse
import os
import warnings
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
    BitsAndBytesConfig)
import bitsandbytes as bnb
from transformers.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from peft import PeftModel
from awq import AutoAWQForCausalLM
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


def to_native(value):
    if isinstance(value, ListConfig):
        return list(value)
    return value


def pred_postprocess_chat(preds: list[list]) -> list[str]:
    result = []
    for el in preds:
        pred = el.split('\n')[-1].strip()
        if pred in label2id:
            pred = label2id[pred]
        else:
            print(f'Strange pred: {pred}')
            pred = 0
        result.append(pred)
    return result


def pred_postprocess(preds: list[list]) -> list[str]:
    result = []
    for el in preds:
        text = ''.join(el).replace('\n', '')
        pred = text.split('|||')[-1].strip()
        if pred in label2id:
            pred = label2id[pred]
        else:
            print(f'Strange pred: {pred}')
            pred = 0
        result.append(pred)
    return result


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

            if data_config.collator_type == 'chat':
                preds = pred_postprocess_chat(preds)
                target_texts = [label2id[el.strip()] for el in batch_inference["labels"]]
            else:
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
    parser.add_argument('--config_id', type=str, required=True)
    args = parser.parse_args()

    initialize(config_path='../../inference_configs')
    data_config = compose(config_name=args.config_id)['data_config']
    model_config = compose(config_name=args.config_id)['model_config']

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision('high')
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.experiment_path)

    print("EOS Token:", tokenizer.eos_token)
    print("EOS Token ID:", tokenizer.eos_token_id)
    print("PAD Token:", tokenizer.pad_token)
    print("PAD Token ID:", tokenizer.pad_token_id)

    if 'cls' in data_config.task_type:

        preprocessor = ClsPreprocessor(
            tokenizer,
            data_config.prefix,
            data_config.max_input_len)

        test_data = read_csv_as_dicts(os.path.join(args.data_set_path, 'test.csv'))
        print(test_data[0])

        if args.limit:
            test_data = test_data[:50]

        print('test_data: ', len(test_data))

        if data_config.collator_type == 'chat':
            test_data = [preprocessor.transform_example_to_chat_inference(el) for el in test_data]
        else:
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
    
    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    bnb_config = None

    if model_config.quant8:
      bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    if model_config.quant4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch_dtype)
        
    print('bnb_config: ', bnb_config)

    if model_config.awq:
        model = AutoAWQForCausalLM.from_quantized(
            args.experiment_path,
            device_map="auto",
            fuse_layers=model_config.fuse_layers,
            use_flash_attn=model_config.use_flash_attn)
        
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.experiment_path,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            token=model_config.token,
            attn_implementation='eager',
            use_cache=False)
    
    if model_config.peft:

        model = PeftModel.from_pretrained(model, args.experiment_path)
        
        if 'T-lite' in args.experiment_path:
            model.generation_config.pad_token_id = model.generation_config.eos_token_id

    print('Model [X]')

    label2id = {
        'да': 1,
        'нет': 0,
    }

    test_dataset = Dataset(test_data)
    
    inference_dataloader = DataLoader(
        test_dataset,
        batch_size=data_config.per_device_inference_batch_size,
        shuffle=False,
        collate_fn=data_collator.collate_fn_inference)
    
    metrics = run_evaluation(
        peft_model=model,
        inference_dataloader=inference_dataloader,
        tokenizer=tokenizer,
        label2id=label2id,
        data_config=data_config,
        device=device)
    
    del (model, inference_dataloader, test_dataset)
    
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    
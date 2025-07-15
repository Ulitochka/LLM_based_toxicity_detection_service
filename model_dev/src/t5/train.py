import os
import time
import argparse
from uuid import uuid4
import warnings
import csv

import torch
import wandb
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments)
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from hydra.experimental import compose, initialize
from transformers.utils.logging import disable_progress_bar

from data_tools.cls_preprocessor import ClsPreprocessor
from metrics.metrics_cls import ClsMetrics
from data_tools.data_collator import DataCollator

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
# disable_progress_bar()

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Trainer.tokenizer is now deprecated.*")


def read_csv_as_dicts(path, delimiter=',') -> list:
    data = []
    with open(path) as f:
        lines = csv.DictReader(f, delimiter=delimiter)
        for row in lines:
            data.append(row)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True)
    parser.add_argument('--data_set_path', type=str, required=True)
    parser.add_argument('--limit', action='store_true', required=False)
    parser.add_argument('--wandb', action='store_true', required=False)
    parser.add_argument('--config_id', type=str, required=True)
    args = parser.parse_args()

    initialize(config_path='../tune_configs')
    train_config = compose(config_name=args.config_id)['train_config']
    optimizer_config = compose(config_name=args.config_id)['optimizer_config']
    model_config = compose(config_name=args.config_id)['model_config']
    data_config = compose(config_name=args.config_id)['data_config']

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    experiment_id = time.strftime(f'%Y_%m_%d-%H_%M_%S') + str(uuid4())[:8] + f"_{model_config.model.replace('/', '_')}"
    new_res_path = os.path.join(args.experiment_path, f"{experiment_id}")
    if not os.path.exists(new_res_path):
        os.makedirs(new_res_path)

    train_args = Seq2SeqTrainingArguments(output_dir="./default")
    for el in [('train_config', train_config), ('optimizer_config', optimizer_config)]:
        for key, value in el[1].items():
            if hasattr(train_args, key):
                setattr(train_args, key, value)
                print(f"Установлено: from {el[0]} {key} = {value}")
            else:
                print(f"Пропущено:  from {el[0]} {key} — не найден в Seq2SeqTrainingArguments")

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

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_config.model)

    collator = DataCollator(tokenizer)

    if 'cls' in data_config.task_type:

        metrics = ClsMetrics(
            tokenizer=tokenizer,
            res_path=new_res_path,
            report_to=train_args.report_to)

        preprocessor = ClsPreprocessor(
            tokenizer=tokenizer,
            prefix=data_config.prefix,
            max_input_length=data_config.max_input_len,
            max_target_length=data_config.max_target_len)
        
        train_data = read_csv_as_dicts(os.path.join(args.data_set_path, 'train.csv'))
        test_data = read_csv_as_dicts(os.path.join(args.data_set_path, 'test.csv'))
        print(test_data[0])

        print('train_data: ', len(train_data))
        print('test_data: ', len(test_data))

        if args.limit:
            test_data = test_data[:200]
            train_data = train_data[:2000]

        train = [preprocessor.preprocess(el) for el in train_data]
        test = [preprocessor.preprocess(el) for el in test_data]
        
    print('Data [X]')

    model = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=model_config.model,
        use_cache=train_config['use_cache'])
    model.config.max_length = data_config.max_target_len
    model.to(device)

    trainer = Seq2SeqTrainer(
        model=model,
        train_dataset=train,
        eval_dataset=test,
        data_collator=collator,
        compute_metrics=metrics.compute_metrics,
        args=train_args)

    trainer.train()
    wandb.finish()

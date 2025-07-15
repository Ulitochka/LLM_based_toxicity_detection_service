import os

import numpy as np
import wandb
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score


class ClsMetrics:
    def __init__(self, tokenizer: AutoTokenizer, res_path: str, report_to: str):
        self._tokenizer = tokenizer
        self._res_path = res_path
        self._count_debug = 0
        self._report_to = report_to

        self._id2labels = {
            1: '1.0',
            0: '0.0',
        }

    def write_text_file(self, path: str, data: list, m: str) -> None:
        with open(os.path.join(path), m) as f:
            for el in data:
                f.write(el + '\n')

    def compute_metrics(self, model_out) -> dict:
        labels = model_out.label_ids
        predictions = model_out.predictions.argmax(-1)

        target_metrics = {
            "f1": f1_score(labels, predictions),
            "precision": precision_score(labels, predictions),
            "recall": recall_score(labels, predictions)
        }

        if self._report_to:
            wandb.log(target_metrics)

        return target_metrics

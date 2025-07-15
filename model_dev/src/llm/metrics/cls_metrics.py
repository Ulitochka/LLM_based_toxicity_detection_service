import os
import re
import string

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

        self._label2id = {
            'да': 1,
            'нет': 0,
        }

    def write_text_file(self, path: str, data: list, m: str) -> None:
        with open(os.path.join(path), m) as f:
            for el in data:
                f.write(el + '\n')

    def fix_mixed_text(self, text: str) -> str:
        return ''.join(self._replacements.get(char, char) for char in text)

    def preproc_string(self, text: str) -> str:
        return self.fix_mixed_text(
                self.remove_double_spaces(
                    self.remove_punctuation(text)))
    
    def remove_punctuation(self, text: str) -> str:
        return text.translate(str.maketrans('', '', string.punctuation)).strip()
    
    def remove_double_spaces(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()
    
    def tokenize(self, text: str) -> set:
        return set(text.lower().split())

    def compute_metrics(self, model_out) -> dict:

        golden_tokens = model_out.label_ids
        golden_tokens = np.where(golden_tokens != -100, golden_tokens, self._tokenizer.pad_token_id)
        golden_texts = self._tokenizer.batch_decode(golden_tokens, skip_special_tokens=True)
        print(golden_texts[:10])
        golden_texts = [self._label2id[el] for el in golden_texts]

        predicted_tokens = model_out.predictions
        predicted_texts = self._tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)
        print(predicted_texts[:10])
        predicted_texts = [self._label2id[el] if el in self._label2id else 0 for el in predicted_texts]

        target_metrics = {
            "f1": f1_score(golden_texts, predicted_texts),
            "precision": precision_score(golden_texts, predicted_texts),
            "recall": recall_score(golden_texts, predicted_texts)
        }

        if self._report_to:
            wandb.log(target_metrics)

        self._count_debug += 1

        return target_metrics

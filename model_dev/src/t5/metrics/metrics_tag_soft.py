import os
import re
import string

import numpy as np
import wandb
from tqdm import tqdm
from transformers import AutoTokenizer

from collections import defaultdict


class TagMetricsSoft:
    def __init__(self, tokenizer: AutoTokenizer, join_char: str, res_path: str, report_to: str):
        self._tokenizer = tokenizer
        self._join_char = join_char
        self._replacements = {'a': 'а', 'c': 'с', 'e': 'е', 'o': 'о', 'p': 'р', 'x': 'х'}
        self._res_path = res_path
        self._count_debug = 0
        self._report_to = report_to

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

    def match_facts(self, gold_facts: set, pred_facts: set, text: str) -> tuple:
        matched_gold = set()
        matched_pred = set()

        for g in gold_facts:
            for p in pred_facts:
                if g in p or p in g:
                    matched_gold.add(g)
                    matched_pred.add(p)

        tp = len(matched_gold)
        # Новый фильтр: исключаем из fp предсказанные факты, которые >= 2 символов и есть в тексте
        real_fp = {fact for fact in (pred_facts - matched_pred) if not (len(fact) >= 2 and fact in text)}
        fp = len(real_fp)
        fn = len(gold_facts - matched_gold)

        return tp, fp, fn
    
    def calculate(self, golden_texts: list, predicted_texts: list, categories: list, prefix: bool, texts: list) -> dict:

        total_tp = 0
        total_fp = 0
        total_fn = 0
        cat_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        errors = []

        for gold, pred, category, text in tqdm(zip(golden_texts, predicted_texts, categories, texts), total=len(golden_texts)):

            is_negative = (gold.strip().lower() == 'нет')

            if (prefix == 'positive' and not is_negative) or (prefix == 'negative' and is_negative):
      
                gold_facts = gold.split(' I ')
                pred_facts = pred.split(' I ')

                gold_facts = set([self.preproc_string(el) for el in gold_facts])
                pred_facts = set([self.preproc_string(el) for el in pred_facts])

                if gold_facts != pred_facts:
                    errors.append(f'gold={str(gold_facts)};pred_facts={str(pred_facts)};text={text}')

                if prefix == 'positive':
                    tp, fp, fn = self.match_facts(gold_facts, pred_facts, text)
                else:
                    tp = len(gold_facts & pred_facts)
                    fp = len(pred_facts - gold_facts)
                    fn = len(gold_facts - pred_facts)

                total_tp += tp
                total_fp += fp
                total_fn += fn

                cat_stats[category]["tp"] += tp
                cat_stats[category]["fp"] += fp
                cat_stats[category]["fn"] += fn
            
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall= total_tp / (total_tp+ total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f'[OVERALL{prefix}] Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')

        target_metrics = {
            f"precision_{prefix}": precision * 100,
            f"recall_{prefix}": recall * 100,
            f"f1_{prefix}": f1 * 100}

        for cat, stats in cat_stats.items():
            c_tp, c_fp, c_fn = stats["tp"], stats["fp"], stats["fn"]
            c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0
            c_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0
            c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0
            print(f'[{prefix}_{cat}] Precision: {c_prec:.3f}, Recall: {c_rec:.3f}, F1: {c_f1:.3f}')
            target_metrics[f"{prefix}_{cat}/precision"] = c_prec * 100
            target_metrics[f"{prefix}_{cat}/recall"] = c_rec * 100
            target_metrics[f"{prefix}_{cat}/f1"] = c_f1 * 100

        if self._res_path:
            self.write_text_file(
                f'{self._res_path}/{prefix}_errors_{self._count_debug}.txt',
                sorted(set(errors)),
                'w')

        return target_metrics

    def compute_metrics(self, model_out) -> dict:

        golden_tokens = model_out.label_ids
        golden_tokens = np.where(golden_tokens != -100, golden_tokens, self._tokenizer.pad_token_id)
        golden_texts = self._tokenizer.batch_decode(golden_tokens, skip_special_tokens=True)

        predicted_tokens = model_out.predictions
        predicted_texts = self._tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)

        inputs = model_out.inputs
        inputs = np.where(inputs != -100, inputs, self._tokenizer.pad_token_id)
        texts = self._tokenizer.batch_decode(inputs, skip_special_tokens=True)
        categories = [t.lower().split(' ? ')[0].replace('есть что о ', '') for t in texts]

        # Среднее по токенам (микро-усреднение), будет учитывать вес каждого токена, независимо от того, 
        # в каком тексте он был.

        print('Sample_preds:', [golden_texts[2]], [predicted_texts[2]], categories[2])

        target_metrics_positive = self.calculate(golden_texts, predicted_texts, categories, 'positive', texts)
        target_metrics_negative = self.calculate(golden_texts, predicted_texts, categories, 'negative', texts)
        target_metrics = {**target_metrics_positive, **target_metrics_negative}

        if self._report_to:
            wandb.log(target_metrics)

        self._count_debug += 1

        return target_metrics

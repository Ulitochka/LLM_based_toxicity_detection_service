import json

from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer


class SynthPreprocessor:
    def __init__(
            self,
            tokenizer: AutoTokenizer,
            max_input_length: int,
            max_target_length: int,
            join_char: str,
            max_negative_examples: int):

        self._tokenizer = tokenizer
        self._max_target_length = max_target_length
        self._max_input_length = max_input_length
        self._join_char=join_char
        self._max_negative_examples = max_negative_examples

    def load_json_per_string(self, path2data: str):
        with open(path2data) as f:
            for line in f:
                yield json.loads(line.rstrip())

    def load_data(self, path: str) -> list:
        results = []
        data = self.load_json_per_string(path)
        for el in data:
            results.append([el['init_text'], el['labels']])
        return results

    def preprocess(self, data: list, data_name: str) -> dict:

        text_inputs = []
        target_texts = []
        result = []

        for el in data:
            count_nagative = 0
            for cat in el[1]:
                if el[1][cat]:
                    text_inputs.append(f'есть что о {cat.lower()} ? {el[0].lower()}')
                    target_texts.append(f' {self._join_char} '.join([f.lower() for f in el[1][cat]]))
                else:
                    if count_nagative < self._max_negative_examples:
                        text_inputs.append(f'есть что о {cat.lower()} ? {el[0].lower()}')
                        target_texts.append('нет')
                        count_nagative += 1

        print('Sample: ', [text_inputs[0], target_texts[0]])

        for i in tqdm(range(len(text_inputs))):
            model_inputs = self._tokenizer(text_inputs[i], max_length=self._max_input_length, padding=False, truncation=True)
            model_inputs['labels'] = self._tokenizer(
                target_texts[i], max_length=self._max_target_length, padding=False, truncation=True)['input_ids']
            # print(model_inputs['labels'], target_texts[i])
            result.append(model_inputs)

        return result

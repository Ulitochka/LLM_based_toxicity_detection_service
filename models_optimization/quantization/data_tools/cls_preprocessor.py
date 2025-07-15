from transformers import AutoTokenizer
import torch


class ClsPreprocessor:
    def __init__(self, tokenizer: AutoTokenizer, prefix: str, max_input_len: int):
        self._tokenizer = tokenizer
        self._prefix = prefix
        self._max_input_len = max_input_len

        self._id2label = {
            '1.0': 'да',
            '0.0': 'нет',
        }

    def transform_example(self, example: list) -> dict:
        input_text = f'{self._prefix} {example["comment"].lower()}'
        target_text = self._id2label[example["toxic"]]
        text = f'{input_text} ||| {target_text} {self._tokenizer.eos_token}'
        return text

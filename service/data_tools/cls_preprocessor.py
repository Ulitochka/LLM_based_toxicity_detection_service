from transformers import AutoTokenizer


class ClsPreprocessor:
    def __init__(self, tokenizer: AutoTokenizer, prefix: str, max_input_len: int):
        self._tokenizer = tokenizer
        self._prefix = prefix
        self._max_input_len = max_input_len

        self._id2label = {
            '1.0': 'да',
            '0.0': 'нет',
        }

    def transform_example(self, example: dict) -> dict:
        input_text = f'{self._prefix} {example["comment"].lower()} |||'
        target_text = self._id2label[example["toxic"]]
        return {'comment': input_text, 'toxic': target_text}

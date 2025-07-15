

class ClsPreprocessor:
    def __init__(
            self,
            tokenizer,
            max_input_length):

        self._tokenizer = tokenizer
        self._max_input_length = max_input_length

        self._label2id = {
            '1.0': 1,
            '0.0': 0,
        }

    def preprocess(self, example: dict) -> dict:
        model_inputs = self._tokenizer(example["comment"].lower(), max_length=self._max_input_length, padding=False, truncation=True)
        label = self._label2id[example["toxic"]]
        model_inputs["labels"] = label
        return model_inputs
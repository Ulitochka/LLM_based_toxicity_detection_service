

class ClsPreprocessor:
    def __init__(
            self,
            tokenizer,
            max_input_length,
            max_target_length,
            prefix):

        self._tokenizer = tokenizer
        self._max_target_length = max_target_length
        self._max_input_length = max_input_length
        self._prefix = prefix

        self._id2label = {
            '1.0': 'да',
            '0.0': 'нет',
        }

    def preprocess(self, example: dict) -> dict:
        input = self._prefix + example["comment"].lower()
        model_inputs = self._tokenizer(input, max_length=self._max_input_length, truncation=True)
        labels = self._tokenizer(text_target=self._id2label[example["toxic"]], max_length=self._max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

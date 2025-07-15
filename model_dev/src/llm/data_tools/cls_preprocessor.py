from transformers import AutoTokenizer


class ClsPreprocessor:
    def __init__(self, tokenizer: AutoTokenizer, prefix: str, max_len: int):
        self._tokenizer = tokenizer
        self._prefix = prefix
        self._max_len = max_len

        self._id2label = {
            '1.0': 'да',
            '0.0': 'нет',
        }

        self._promt = """Это текст комментария из социальной сети': {} Определи содержит ли он оскоробления, мат, токсичность."""

    def transform_example(self, example: dict) -> dict:
        input_text = f'{self._prefix} {example["comment"].lower()}'
        target_text = self._id2label[example["toxic"]]
        return {"text": input_text, "label": target_text}
    
    def truncate_input_to_fit_limit(self, input_text: str, target_text: str, mode: str) -> str:
        """
        Усечение input_text по токенам так, чтобы итоговая длина chat-template
        с assistant-ответом не превышала max_tokens.
        """
        # Строим полный row_json
        full_row = [
                {"role": "user", "content": self._promt.format(input_text)},
                {"role": "assistant", "content": target_text}]

        if mode == 'inference':
            full_row = [{"role": "user", "content": self._promt.format(input_text)}]

        # Токенизируем для оценки длины
        tokenized = self._tokenizer.apply_chat_template(
            full_row,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        )
        total_len = tokenized.shape[-1]

        if total_len <= self._max_len:
            return input_text  # ничего не усекать

        # Вычисляем, на сколько токенов превышает лимит
        excess = total_len - self._max_len

        # Получаем токены только от user части
        user_tokens = self._tokenizer(
            input_text,
            truncation=False,
            add_special_tokens=False
        )["input_ids"]

        # Усечение с конца
        if excess >= len(user_tokens):
            trimmed_tokens = []
        else:
            trimmed_tokens = user_tokens[:-excess]

        # Обратно в текст
        trimmed_text = self._tokenizer.decode(
            trimmed_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return trimmed_text

    def transform_example_to_chat(self, example: dict) -> list:
        input_text = example["comment"].lower()
        target_text = self._id2label[example["toxic"]]

        trimmed_text = self.truncate_input_to_fit_limit(input_text, target_text)

        row_json = [
            {"role": "user", "content": self._promt.format(trimmed_text)},
            {"role": "assistant", "content": target_text}]
        
        full_prompt = self._tokenizer.apply_chat_template(
            row_json,
            tokenize=False,
            add_generation_prompt=False)

        prefix_prompt = self._tokenizer.apply_chat_template(
            [row_json[0]],
            add_generation_prompt=True, #  токенизатор добавляет финальную часть "<|assistant|>"
            tokenize=False)
        
        prefix_inputs = self._tokenizer(
            text=prefix_prompt,
            return_tensors="pt",
            truncation=False,
            padding=False)

        prefix_len = prefix_inputs.input_ids.shape[-1]

        return {"full_prompt": full_prompt, "prefix_len": prefix_len}
    
    def transform_example_to_chat_inference(self, example: dict) -> list:
        input_text = example["comment"].lower()
        target_text = self._id2label[example["toxic"]]

        trimmed_text = self.truncate_input_to_fit_limit(input_text, target_text, 'inference')

        row_json = [{"role": "user", "content": self._promt.format(trimmed_text)}]
        
        full_prompt = self._tokenizer.apply_chat_template(
            row_json,
            tokenize=False,
            add_generation_prompt=True)
        
        return {"full_prompt": full_prompt, "target_text": target_text}

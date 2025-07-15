import torch


class DataCollator:
    def __init__(self, tokenizer, max_input_len, max_target_len):
        self._tokenizer = tokenizer
        self._max_input_len = max_input_len
        self._max_target_len = max_target_len

    def pad_seq(self, seq: list[int], max_batch_len: int, pad_value: int) -> list[int]:
        return (max_batch_len - len(seq)) * [pad_value] + seq

    def collate_fn(self, batch: list) -> dict:
        batch_attention_masks = list()
        batch_inputs = list()
        batch_labels = list()

        texts = [f'{sample["text"]} ||| {sample["label"]} {self._tokenizer.eos_token}' for sample in batch]
        labels = [f'{sample["label"]} {self._tokenizer.eos_token}' for sample in batch]

        model_inputs = self._tokenizer(texts, padding=False, max_length=self._max_input_len, add_special_tokens=False, truncation=True)
        model_inputs['labels'] = self._tokenizer(labels, padding=False, add_special_tokens=False)['input_ids']
        max_size = max([len(ex) for ex in model_inputs['input_ids']])

        for i in range(len(texts)):
            batch_inputs += [self.pad_seq(model_inputs['input_ids'][i], max_size, self._tokenizer.eos_token_id)]
            attention_mask_item = [1 for i in range(len(model_inputs['input_ids'][i]))]
            batch_attention_masks += [self.pad_seq(attention_mask_item, max_size, 0)]
            batch_labels += [self.pad_seq(model_inputs['labels'][i], max_size, -100)]

        return {
            "input_ids": torch.tensor(batch_inputs, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_masks, dtype=torch.long),
            "labels": torch.tensor(batch_labels)}
    
    def prepare_generation(self, batch: list) -> dict:
        batch_attention_masks_inference = list()
        batch_inputs_inference = list()

        texts_inference = [f'{sample["text"]} |||' for sample in batch]
        model_inputs = self._tokenizer(texts_inference, padding=False, add_special_tokens=False)
        max_size = max([len(ex) for ex in model_inputs['input_ids']])

        for i in range(len(texts_inference)):
            batch_inputs_inference += [self.pad_seq(model_inputs['input_ids'][i], max_size, self._tokenizer.eos_token_id)]
            attention_mask_item_inference = [1 for i in range(len(model_inputs['input_ids'][i]))]
            batch_attention_masks_inference += [self.pad_seq(attention_mask_item_inference, max_size, 0)]

        result = {
            "input_ids_inference": torch.tensor(batch_inputs_inference, dtype=torch.long),
            "attention_mask_inference": torch.tensor(batch_attention_masks_inference, dtype=torch.long)}
        
        return result

    def collate_fn_inference(self, batch: list) -> dict:
        batch_inference = self.prepare_generation(batch)
        batch_train = self.collate_fn(batch)        
        result = {**batch_inference, **batch_train}
        return result

import torch

from itertools import takewhile


class DataCollatorChat:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def collate_fn(self, examples: list):
        prompts_batch = []
        prefix_lengths = []

        for el in examples:
            prompts_batch.append(el['full_prompt'])
            prefix_lengths.append(el['prefix_len'])

        batch = self._tokenizer(
            text=prompts_batch,
            return_tensors="pt",
            truncation=True,
            padding="longest")  
        
        new_prefix_lengths = self.update_prefix_len(batch["input_ids"], prefix_lengths)

        labels = batch["input_ids"].clone()
        labels[labels == self._tokenizer.pad_token_id] = -100
        for batch_idx in range(len(examples)):
            labels[batch_idx, :new_prefix_lengths[batch_idx]] = -100
        batch["labels"] = labels

        return batch
    
    def update_prefix_len(self, input_ids, prefix_lengths):
        pad_token_id = self._tokenizer.pad_token_id
        new_prefix_lengths = [
            prefix_len + sum(1 for _ in takewhile(lambda x: x == pad_token_id, input_id))
            for input_id, prefix_len in zip(input_ids, prefix_lengths)]
        return new_prefix_lengths
    
    def collate_fn_inference(self, examples: list) -> dict:
        prompts_batch = []
        target_texts = []

        for el in examples:
            prompts_batch.append(el['full_prompt'])
            target_texts.append(el['target_text'])

        batch = self._tokenizer(
            text=prompts_batch,
            return_tensors="pt",
            truncation=True,
            padding="longest")

        result = {
            "input_ids_inference": batch['input_ids'],
            "attention_mask_inference": batch['attention_mask'],
            "labels": target_texts}

        return result

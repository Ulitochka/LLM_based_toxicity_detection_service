import torch


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def pad_seq(self, seq: list[int], max_batch_len: int, pad_value: int) -> list[int]:
        return seq + (max_batch_len - len(seq)) * [pad_value]

    def __call__(self, batch: list) -> dict:
        batch_attention_masks = list()
        batch_inputs = list()
        batch_labels = list()

        max_size_input = max([len(ex['input_ids']) for ex in batch])
        max_size_target = max([len(ex['labels']) for ex in batch])
        for item in batch:
            batch_inputs += [self.pad_seq(item['input_ids'], max_size_input, self.tokenizer.pad_token_id)]
            attention_mask_item = [1 for i in range(len(item['input_ids']))]
            batch_attention_masks += [self.pad_seq(attention_mask_item, max_size_input, 0)]
            batch_labels += [self.pad_seq(item['labels'], max_size_target, -100)]

        result = {
            "input_ids": torch.tensor(batch_inputs, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_masks, dtype=torch.long),
            "labels":  torch.tensor(batch_labels, dtype=torch.long)
        }

        return result

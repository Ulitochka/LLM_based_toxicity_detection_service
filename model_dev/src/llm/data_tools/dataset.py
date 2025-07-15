from transformers import AutoProcessor
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(
            self,
            samples: list):

        self._samples = samples
    
    def __getitem__(self, index: int):
        return self._samples[index]

    def __len__(self) -> int:
        return len(self._samples)

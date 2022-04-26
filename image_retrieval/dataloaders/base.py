import numpy as np
from PIL import Image
from abc import abstractmethod

import torch
from torch.utils.data import Dataset


class ImageRetrievalDataset(Dataset):
    def __init__(self, artifact_id: str, tokenizer=None, max_length: int = 100) -> None:
        super().__init__()
        self.artifact_id = artifact_id
        self.image_files, self.captions = self.fetch_dataset()
        self.tokenized_captions = tokenizer(
            self.captions, padding=True, truncation=True, max_length=max_length
        )

    @abstractmethod
    def fetch_dataset(self):
        pass

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        image = np.array(Image.open(self.image_files[index]))
        image = torch.tensor(image).permute(2, 0, 1).float()
        caption = self.captions[index]
        input_ids = torch.tensor(self.tokenized_captions["input_ids"])
        attention_mask = torch.tensor(self.tokenized_captions["attention_mask"])
        return image, caption, input_ids, attention_mask

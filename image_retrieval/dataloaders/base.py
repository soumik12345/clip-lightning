import numpy as np
from PIL import Image
from typing import Optional
from abc import abstractmethod

import torch
from torch.utils.data import Dataset


class ImageRetrievalDataset(Dataset):
    def __init__(
        self,
        artifact_id: str,
        tokenizer=None,
        target_size: Optional[int] = None,
        max_length: int = 100,
    ) -> None:
        super().__init__()
        self.artifact_id = artifact_id
        self.target_size = target_size
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
        image = Image.open(self.image_files[index])
        image = (
            image.resize((self.target_size, self.target_size))
            if self.target_size is not None
            else image
        )
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float()
        caption = self.captions[index]
        input_ids = torch.tensor(self.tokenized_captions["input_ids"])
        attention_mask = torch.tensor(self.tokenized_captions["attention_mask"])
        return image, caption, input_ids, attention_mask

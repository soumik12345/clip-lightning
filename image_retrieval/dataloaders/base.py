import numpy as np
from tqdm import tqdm
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
        lazy_loading: bool = False,
    ) -> None:
        super().__init__()
        self.artifact_id = artifact_id
        self.target_size = target_size
        image_files, self.captions = self.fetch_dataset()
        self.lazy_loading = lazy_loading
        self.images = (
            [self.read_image(idx) for idx in tqdm(range(len(self)))]
            if lazy_loading
            else image_files
        )
        assert tokenizer is not None
        self.tokenized_captions = tokenizer(
            self.captions, padding=True, truncation=True, max_length=max_length
        )

    def read_image(self, index):
        image = Image.open(self.image_files[index])
        image = (
            image.resize((self.target_size, self.target_size))
            if self.target_size is not None
            else image
        )

    @abstractmethod
    def fetch_dataset(self):
        pass

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        image = self.read_image(image) if not self.lazy_loading else self.images[index]
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float()
        caption = self.captions[index]
        input_ids = torch.tensor(self.tokenized_captions["input_ids"])[index]
        attention_mask = torch.tensor(self.tokenized_captions["attention_mask"])[index]
        return image, caption, input_ids, attention_mask

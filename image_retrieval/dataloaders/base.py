from abc import abstractmethod
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageRetrievalDataset(Dataset):
    def __init__(
        self,
        artifact_id: str,
        tokenizer=None,
        target_size: Optional[int] = None,
        max_length: int = 200,
        lazy_loading: bool = False,
        mask: bool = False
    ) -> None:
        super().__init__()
        self.artifact_id = artifact_id
        self.target_size = target_size
        self.image_files, self.captions = self.fetch_dataset()
        self.lazy_loading = lazy_loading
        self.images = self.image_files
        self.mask = mask

        assert tokenizer is not None

        self.tokenizer = tokenizer

        self.tokenized_captions = tokenizer(
            list(self.captions), 
            padding=True, 
            truncation=True, 
            max_length=max_length,
            return_tensors='pt'
        )
        self.transforms = A.Compose(
            [
                A.Resize(target_size, target_size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )


    def read_image(self, image_file):
        image = Image.open(image_file)
        # image = (
        #     image.resize((self.target_size, self.target_size))
        #     if self.target_size is not None
        #     else image
        # )
        return image

    @abstractmethod
    def fetch_dataset(self):
        pass

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        item = {
            key: values[index]
            for key, values in self.tokenized_captions.items()
        }

        if self.mask:
            c = item["input_ids"].clone().detach()

            rand = torch.rand(c.shape)
            mask_arr = rand < 0.15

            mask_arr = (rand < 0.15) * (item["input_ids"] != self.tokenizer.cls_token_id) * (item["input_ids"] != self.tokenizer.sep_token_id)
            selection = torch.flatten((mask_arr[0]).nonzero()).tolist()

            item["input_ids"][selection] = self.tokenizer.mask_token_id

        image = cv2.imread(self.image_files[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        item["image"] = torch.tensor(image).permute(2, 0, 1).float()
        item["caption"] = self.captions[index]
        return item

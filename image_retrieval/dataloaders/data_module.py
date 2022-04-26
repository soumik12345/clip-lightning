from typing import Optional

from torch.utils.data import random_split, DataLoader
from pytorch_lightning import LightningDataModule

from .base import ImageRetrievalDataset
from .flickr8k import Flickr8kDataset
from .flickr30k import Flickr30kDataset


DATASET_LOOKUP = {"flickr8k": Flickr8kDataset, "flickr30k": Flickr30kDataset}


class ImageRetrievalDataModule(LightningDataModule):
    def __init__(
        self,
        artifact_id: str,
        tokenizer=None,
        target_size: Optional[int] = None,
        max_length: int = 100,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.artifact_id = artifact_id
        self.tokenizer = tokenizer
        self.target_size = target_size
        self.max_length = max_length
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

    def split_data(self, dataset: ImageRetrievalDataset, val_split: float):
        train_length = (1 - val_split) * len(dataset)
        val_length = len(dataset) - train_length
        train_dataset, val_dataset = random_split(
            dataset, lengths=[train_length, val_length]
        )
        return train_dataset, val_dataset

    def setup(
        self,
        stage: Optional[str] = None,
        dataset_name: str = "flickr8k",
        val_split: float = 0.2,
    ) -> None:
        dataset = DATASET_LOOKUP[dataset_name](
            artifact_id=self.artifact_id,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        self.train_dataset, self.val_dataset = self.split_data(
            dataset, val_split=val_split
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size)

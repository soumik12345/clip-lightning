from typing import Optional

from torch.utils.data import random_split, DataLoader
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer

from .base import ImageRetrievalDataset
from .flickr8k import Flickr8kDataset
from .flickr30k import Flickr30kDataset
from .utils import collate_fn


DATASET_LOOKUP = {"flickr8k": Flickr8kDataset, "flickr30k": Flickr30kDataset}


class ImageRetrievalDataModule(LightningDataModule):
    def __init__(
        self,
        artifact_id: str,
        dataset_name: str,
        val_split: float = 0.2,
        tokenizer_alias: Optional[str] = None,
        target_size: Optional[int] = 224,
        max_length: int = 100,
        lazy_loading: bool = False,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        num_workers: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.artifact_id = artifact_id
        self.dataset_name = dataset_name
        self.val_split = val_split
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_alias)
        self.target_size = target_size
        self.max_length = max_length
        self.lazy_loading = lazy_loading
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

    @staticmethod
    def split_data(dataset: ImageRetrievalDataset, val_split: float):
        train_length = int((1 - val_split) * len(dataset))
        val_length = len(dataset) - train_length
        train_dataset, val_dataset = random_split(
            dataset, lengths=[train_length, val_length]
        )
        return train_dataset, val_dataset

    def setup(
        self,
        stage: Optional[str] = None,
    ) -> None:
        dataset = DATASET_LOOKUP[self.dataset_name](
            artifact_id=self.artifact_id,
            tokenizer=self.tokenizer,
            target_size=self.target_size,
            max_length=self.max_length,
            lazy_loading=self.lazy_loading,
        )
        self.train_dataset, self.val_dataset = self.split_data(
            dataset, val_split=self.val_split
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
        )

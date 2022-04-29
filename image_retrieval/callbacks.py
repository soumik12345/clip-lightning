import wandb
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Any, List, Union

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from .dataloaders import ImageRetrievalDataset
from .models import CLIPDualEncoderModel


class LogPredictionCallback(Callback):
    def __init__(
        self,
        search_text: str,
        tokenizer,
        max_length: int,
        max_matches: int,
        image_files: List[str],
        validation_dataloader: ImageRetrievalDataset,
    ) -> None:
        super().__init__()
        self.search_text = search_text
        assert tokenizer is not None
        self.tokenized_search_text = tokenizer(
            [search_text], padding=True, truncation=True, max_length=max_length
        )
        self.input_ids = torch.tensor(self.tokenized_captions["input_ids"])[0]
        self.attention_mask = torch.tensor(self.tokenized_captions["attention_mask"])[0]
        self.image_files = image_files
        self.validation_dataloader = validation_dataloader
        self.max_matches = max_matches
        self.data_viz_table: Union[None, wandb.Table] = None

    def get_embeddings(self, pl_module: CLIPDualEncoderModel):
        image_embedding_list = []
        with torch.no_grad():
            for batch in tqdm(self.validation_dataloader):
                image, _, _, _ = batch
                image_features = pl_module.image_encoder(image)
                image_embeddings = pl_module.image_projection(image_features)
                image_embedding_list.append(image_embeddings)
            image_embeddings = torch.cat(image_embedding_list)
            text_features = pl_module.text_encoder(self.input_ids, self.attention_mask)
            text_embeddings = pl_module.text_projection(text_features)
            return image_embeddings, text_embeddings

    def create_table(self, matches, batch_idx: int):
        images = [wandb.Image(np.array(Image.open(match))) for match in matches]
        if self.data_viz_table is not None:
            self.data_viz_table.add_data(batch_idx, self.search_text, images)
        else:
            self.data_viz_table = wandb.Table(
                columns=["Batch Index", "Search Query", "Images"],
                data=[batch_idx, self.search_text, images],
            )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: CLIPDualEncoderModel,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        image_embeddings, text_embeddings = self.get_embeddings(pl_module)
        image_embeddings_normalized = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings_normalized = F.normalize(text_embeddings, p=2, dim=-1)
        dot_similarity = text_embeddings_normalized @ image_embeddings_normalized.T
        values, indices = torch.topk(dot_similarity.squeeze(0), self.max_matches * 5)
        matches = [self.image_files[idx] for idx in indices[::5]]
        self.create_table(matches, batch_idx)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: CLIPDualEncoderModel
    ) -> None:
        self.log({"Predictions": self.data_viz_table})

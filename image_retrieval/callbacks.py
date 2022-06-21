from typing import Any, List, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning.callbacks import Callback
from tqdm import tqdm
from transformers import AutoTokenizer

from models import CLIPDualEncoderModel


class LogPredictionCallback(Callback):
    def __init__(
        self,
        tokenizer: str,
        max_length: int = 200,
        max_matches: int = 5,
        max_sentences: int = 20,
    ) -> None:
        super().__init__()
        assert tokenizer is not None
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length
        self.max_matches = max_matches
        self.max_sentences = max_sentences
        self.data_viz_table: Union[None, wandb.Table] = None
        self.validation_dataloader: Union[None, torch.utils.data.DataLoader] = None
        self.search_text = []
        self.images = None

        self.is_setup = False

    def setup_search_text(self, text: List[str]):
        self.tokenized_captions = self.tokenizer(
            text, padding=True, truncation=True, max_length=self.max_length
        )
        self.input_ids = torch.tensor(self.tokenized_captions["input_ids"])
        self.attention_mask = torch.tensor(self.tokenized_captions["attention_mask"])

    def get_embeddings(self, pl_module: CLIPDualEncoderModel):
        image_embedding_list = []
        with torch.no_grad():
            for batch in tqdm(self.validation_dataloader):
                image = batch["image"]
                image = image.to(pl_module.device)
                image_features = pl_module.image_encoder(image)
                image_embeddings = pl_module.image_projection(image_features)
                image_embedding_list.append(image_embeddings)
            image_embeddings = torch.cat(image_embedding_list)
            self.input_ids = self.input_ids.to(pl_module.device)
            self.attention_mask = self.attention_mask.to(pl_module.device)
            text_features = pl_module.text_encoder(self.input_ids, self.attention_mask)
            text_embeddings = pl_module.text_projection(text_features)
            return image_embeddings, text_embeddings

    def create_table(self, matches):
        table = wandb.Table(columns=["Index", "Text", "Matches"])
        for idx, text in enumerate(self.search_text):
            row = [idx, text, [wandb.Image(match) for match in matches[idx]]]
            table.add_data(*row)
        return table

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: CLIPDualEncoderModel,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        if self.validation_dataloader is None:
            self.validation_dataloader = trainer.datamodule.val_dataloader()
        if len(self.search_text) < self.max_sentences:
            self.search_text.extend(batch["caption"])
        elif len(self.search_text) > self.max_sentences:
            self.search_text = self.search_text[: self.max_sentences]

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: CLIPDualEncoderModel
    ) -> None:
        if not self.is_setup:
            self.setup_search_text(self.search_text)
            self.is_setup = True
        if self.images is None:
            if isinstance(self.validation_dataloader.dataset, torch.utils.data.Subset):
                self.images = self.validation_dataloader.dataset.dataset.images
            else:
                self.images = self.validation_dataloader.dataset.images
        image_embeddings, text_embeddings = self.get_embeddings(pl_module)
        image_embeddings_normalized = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings_normalized = F.normalize(text_embeddings, p=2, dim=-1)
        dot_similarity = text_embeddings_normalized @ image_embeddings_normalized.T
        values, indices = torch.topk(dot_similarity.squeeze(0), self.max_matches * 5)

        indices = indices.cpu()
        matches = []

        for idx in indices:
            matches.append([self.images[i] for i in idx[:5]])
        if isinstance(trainer.logger.experiment, wandb.wandb_run.Run):
            trainer.logger.experiment.log({"Predictions": self.create_table(matches)})

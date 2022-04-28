import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from .encoders import ImageEncoder, TextEncoder, ProjectionHead


class CLIPDualEncoderModel(LightningModule):
    def __init__(
        self,
        image_encoder_alias: str,
        text_encoder_alias: str,
        image_encoder_pretrained: bool = True,
        image_encoder_trainable: bool = True,
        text_encoder_trainable: bool = True,
        image_embedding_dims: int = 2048,
        text_embedding_dims: int = 768,
        projection_dims: int = 256,
        dropout: float = 0.0,
        temperature: float = 1.0,
        weight_decay: float = 0.0,
        lr_scheduler_patience: float = 1.0,
        lr_scheduler_factor: float = 0.8,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.image_encoder = ImageEncoder(
            model_name=image_encoder_alias,
            pretrained=image_encoder_pretrained,
            trainable=image_encoder_trainable,
        )
        self.text_encoder = TextEncoder(
            model_name=text_encoder_alias, trainable=text_encoder_trainable
        )
        self.image_projection = ProjectionHead(
            embedding_dim=image_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )
        self.text_projection = ProjectionHead(
            embedding_dim=text_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor

        self.save_hyperparameters()

    def _compute_losses(self, image_embeddings, text_embeddings):
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        images_loss = (-targets.T * self.log_softmax(logits.T)).sum(1)
        texts_loss = (-targets * self.log_softmax(logits)).sum(1)
        return (images_loss + texts_loss) / 2.0

    def forward(self, inputs):
        images, _, input_ids, attention_mask = inputs

        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)

        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        return image_embeddings, text_embeddings

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.lr_scheduler_patience,
            factor=self.lr_scheduler_factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        loss = self._compute_losses(image_embeddings, text_embeddings).mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        loss = self._compute_losses(image_embeddings, text_embeddings).mean()
        self.log("val_loss", loss)
        return loss

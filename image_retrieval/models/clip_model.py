import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.seed import seed_everything

from .encoders import ImageEncoder, ProjectionHead, TextEncoder


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
        head_lr: float = 1e-3,
        image_encoder_lr: float = 1e-4,
        text_encoder_lr: float = 1e-5,
        lr_scheduler_patience: float = 1.0,
        lr_scheduler_factor: float = 0.8,
        seed: int = 42,
        target_size: int = 224,
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

        example_input = torch.rand((1, 3, target_size, target_size)).to(self.device)

        example_out = self.image_encoder(example_input)
        output_dims = example_out.shape[-1]
        # print(output_dims)

        self.image_projection = ProjectionHead(
            embedding_dim=output_dims,
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
        self.head_lr = head_lr
        self.image_encoder_lr = image_encoder_lr
        self.text_encoder_lr = text_encoder_lr
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        seed_everything(seed)
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
        return (images_loss + texts_loss) / 2.0, images_loss, texts_loss

    def forward(self, inputs):
        image_features = self.image_encoder(inputs["image"])
        text_features = self.text_encoder(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )

        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        return image_embeddings, text_embeddings

    def configure_optimizers(self):
        parameters = [
            {"params": self.image_encoder.parameters(), "lr": self.image_encoder_lr},
            {"params": self.text_encoder.parameters(), "lr": self.text_encoder_lr},
            {
                "params": itertools.chain(
                    self.image_projection.parameters(),
                    self.text_projection.parameters(),
                ),
                "lr": self.head_lr,
                "weight_decay": self.weight_decay,
            },
        ]
        optimizer = optim.Adam(parameters, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.lr_scheduler_patience,
            factor=self.lr_scheduler_factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val/loss",
        }

    def training_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        loss, image_loss, text_loss = self._compute_losses(image_embeddings, text_embeddings)
        loss = loss.mean()
        train_loss = self.all_gather(loss)
        self.log("train/loss", train_loss.mean())
        self.log("train/image_loss", image_loss.mean())
        self.log("train/text_loss", text_loss.mean())
        return loss

    def validation_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        loss, image_loss, text_loss = self._compute_losses(image_embeddings, text_embeddings)
        loss = loss.mean()
        val_loss = self.all_gather(loss)
        self.log("val/loss", val_loss.mean())
        self.log("val/image_loss", image_loss.mean())
        self.log("val/text_loss", text_loss.mean())
        return loss

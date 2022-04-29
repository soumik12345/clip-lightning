from tqdm import tqdm

import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback

from .dataloaders.base import ImageRetrievalDataset
from .models.clip_model import CLIPDualEncoderModel


class LogPredictionCallback(Callback):
    def __init__(
        self,
        search_text: str,
        tokenizer,
        max_length: int,
        max_matches: int,
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
        self.validation_dataloader = validation_dataloader
        self.max_matches = max_matches

    def get_embeddings(self, pl_module):
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

    def on_validation_batch_end(
        self,
        trainer,
        pl_module: CLIPDualEncoderModel,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ):
        image_embeddings, text_embeddings = self.get_embeddings(pl_module)
        image_embeddings_normalized = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings_normalized = F.normalize(text_embeddings, p=2, dim=-1)
        dot_similarity = text_embeddings_normalized @ image_embeddings_normalized.T
        values, indices = torch.topk(dot_similarity.squeeze(0), self.max_matches * 5)
        # return values, indices

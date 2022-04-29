from tqdm import tqdm

import torch
from pytorch_lightning.callbacks import Callback


class LogPredictionCallback(Callback):
    def __init__(self, search_text: str, tokenizer, max_length: int) -> None:
        super().__init__()
        self.search_text = search_text
        assert tokenizer is not None
        self.tokenized_search_text = tokenizer(
            search_text, padding=True, truncation=True, max_length=max_length
        )
    
    def get_embeddings(self, batch, pl_module):
        image_embedding_list = []
        with torch.no_grad():
            image, _, _, _ = batch
            image_features = pl_module.image_encoder(image)
            image_embeddings = pl_module.image_projection(image_features)
            image_embedding_list.append(image_embeddings)
        return torch.cat(image_embedding_list)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int, dataloader_idx: int
    ):
        pass

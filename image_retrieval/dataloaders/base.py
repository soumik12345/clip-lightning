import os
import wandb
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class ImageRetrievalDataset(Dataset):
    def __init__(self, artifact_id: str, tokenizer=None, max_length: int = 100) -> None:
        super().__init__()
        self.artifact_id = artifact_id
        self.image_files, self.captions = self._fetch_dataset()
        self.tokenized_captions = tokenizer(
            self.captions, padding=True, truncation=True, max_length=max_length
        )

    def _fetch_dataset(self):
        if wandb.run is None:
            wandb.init()
        artifact = wandb.use_artifact(self.artifact_id, type="dataset")
        artifact_dir = artifact.download()
        annotations = pd.read_csv(os.path.join(artifact_dir, "captions.txt"))
        image_files = [
            os.path.join(artifact_dir, "Images", image_file)
            for image_file in annotations["image"].to_list()
        ]
        for image_file in image_files:
            assert os.path.isfile(image_file)
        captions = annotations["caption"].to_list()
        return image_files, captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        image = np.array(Image.open(self.image_files[index]))
        image = torch.tensor(image).permute(2, 0, 1).float()
        caption = self.captions[index]
        input_ids = torch.tensor(self.tokenized_captions["input_ids"])
        attention_mask = torch.tensor(self.tokenized_captions["attention_mask"])
        return image, caption, input_ids, attention_mask

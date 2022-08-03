import os
import wandb
import pandas as pd
from pycocotools.coco import COCO

from .base import ImageRetrievalDataset

from typing import Optional

class COCODataset(ImageRetrievalDataset):
    def __init__(
        self,
        artifact_id: str,
        tokenizer=None,
        target_size: Optional[int] = None,
        max_length: int = 100,
        lazy_loading: bool = False,
        mask: bool = False
    ) -> None:
        super().__init__(artifact_id, tokenizer, target_size, max_length, lazy_loading, mask)
    
    def fetch_dataset(self):
        # if wandb.run is None:
        #     api = wandb.Api()
        #     artifact = api.artifact(self.artifact_id, type="dataset")
        # else:
        #     artifact = wandb.use_artifact(self.artifact_id, type="dataset")
        # artifact_dir = artifact.download()
        img_folder = "/home/manan_goel/coco/train2014"
        train_annotations_file = f"/home/manan_goel/coco/annotations/captions_train2014.json"
        coco = COCO(train_annotations_file)

        ids = list(coco.anns.keys())
        captions = []
        files = []
        for id in range(len(ids)):
            captions.append(coco.anns[ids[id]]['caption'].lower())
            img_id = coco.anns[ids[id]]['image_id']

            files.append(img_folder + "/" + coco.loadImgs(img_id)[0]['file_name'])
        return files, captions

if __name__ == "__main__":
    COCODataset("random")
import numpy as np
import matplotlib.pyplot as plt

from transformers import BertTokenizer
from image_retrieval.dataloaders import Flickr8kDataset


dataset = Flickr8kDataset(
    artifact_id="geekyrakshit/clip-image-retrieval/flickr-8k:latest",
    tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"),
)
image, caption, input_ids, attention_mask = dataset[0]
plt.imshow(image.permute(1, 2, 0).numpy().astype(np.uint8))
plt.title(caption)
plt.show()

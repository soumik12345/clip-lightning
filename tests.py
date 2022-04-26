import unittest

from transformers import BertTokenizer
from image_retrieval.dataloaders import Flickr8kDataset


class Flickr8KTester(unittest.TestCase):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.target_size = 224
        self.artifact_id = "geekyrakshit/clip-image-retrieval/flickr-8k:latest"
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    def test_dataset(self):
        dataset = Flickr8kDataset(artifact_id=self.artifact_id, target_size=self.target_size, tokenizer=self.tokenizer)
        image, caption, input_ids, attention_mask = dataset[0]
        assert isinstance(caption, str)
        assert image.shape == (3, 224, 224)
        assert input_ids.shape == (40455, 42)
        assert attention_mask.shape == (40455, 42)

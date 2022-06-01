
# <center> CLIP with PyTorch Lightning and W&B </center>
<div align="center">
        <img src="https://i.imgur.com/gb6B4ig.png" width="300" />
        <img src="https://github.com/PyTorchLightning/pytorch-lightning/raw/master/docs/source/_static/images/logo.png" width="300" /> 
</div>

This is a simple lightweight implementation of the [CLIP](https://openai.com/blog/clip/) model by OpenAI in PyTorch Lightning which uses Weights & Biases for experiment tracking, visualizing results, comparing performance of different backbone models and hyperparameter optimization.

## CLIP
CLIP (Contrastive Language–Image Pre-training) builds on a large body of work on zero-shot transfer, natural language supervision, and multimodal learning. CLIP pre-trains an image encoder and a text encoder to predict which images were paired with which texts in our dataset. We then use this behavior to turn CLIP into a zero-shot classifier. We convert all of a dataset’s classes into captions such as “a photo of a dog” and predict the class of the caption CLIP estimates best pairs with a given image.

You can read more about CLIP [here](https://openai.com/blog/clip/) and [here](https://arxiv.org/abs/2103.00020)

## Dataset
This implementation of CLIP supports training on two datasets [Flickr8k](https://forms.illinois.edu/sec/1713398) which contains ~8K images with 5 captions for each image and [Flickr30k](https://aclanthology.org/Q14-1006/) which contains ~30K images with corresponding captions. Both the datasets were logged as W&B artifacts for easy accessibility. We use the [PyTorch Lightning DataModule](./image_retrieval/dataloaders/data_module.py) to handle all our data loading related needs including downloading the data, splitting it into training and validation sets and finally setting up the dataloaders.

## Model
A CLIP model uses a text encoder and an image encoder. This repostiry supports pulling image models from [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) and transformer models from [huggingface transformers](https://github.com/huggingface/transformers). These are then encapsulated in a [LightningModule](./image_retrieval/models/clip_model.py) with appropriate callbacks for model training.
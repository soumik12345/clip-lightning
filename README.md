# CLIP with PyTorch Lightning

This repository consists of an implementation of CLIP in PyTorch Lightning.

To train the model use the following command

```shell
python image_retrieval/cli.py fit --model.image_encoder_alias resnet50 --model.text_encoder_alias bert-base-uncased --data.artifact_id geekyrakshit/clip-image-retrieval/flickr-8k:latest --trainer.accelerator gpu
```

This will train the CLIP model with ResNet50 as the image encoder and BERT base uncased as the text encoder on the data stored as a W&B artifact.
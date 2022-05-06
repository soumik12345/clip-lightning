# CLIP with PyTorch Lightning

This repository consists of an implementation of CLIP in PyTorch Lightning.

To train the model use the following command

```shell
python image_retrieval/cli.py fit \
    --model.image_encoder_alias resnet18 \
    --model.text_encoder_alias albert-base-v2 \
    --data.artifact_id geekyrakshit/clip-image-retrieval/flickr-8k:latest \
    --trainer.accelerator gpu \
    --trainer.devices 2 \
    --data.train_batch_size 32 \
    --model.image_embedding_dims 512 \
    --trainer.max_epochs 5 \
    --trainer.logger WandbLogger \
    --data.dataset_name flickr8k \
    --log_prediction_callback.max_images 40 \
    --trainer.logger.project clip
```

This will train the CLIP model with ResNet50 as the image encoder and BERT base uncased as the text encoder on the data stored as a W&B artifact.
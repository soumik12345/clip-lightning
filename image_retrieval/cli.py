from pytorch_lightning.utilities import cli

from dataloaders.data_module import ImageRetrievalDataModule
from models.clip_model import CLIPDualEncoderModel
from callbacks import LogPredictionCallback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor


class CLI(cli.LightningCLI):
    def add_arguments_to_parser(self, parser: cli.LightningArgumentParser) -> None:
        parser.link_arguments("model.text_encoder_alias", "data.tokenizer_alias")
        parser.add_lightning_class_args(
            LogPredictionCallback, "log_prediction_callback"
        )
        parser.link_arguments(
            "model.text_encoder_alias", "log_prediction_callback.tokenizer"
        )
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.add_lightning_class_args(LearningRateMonitor, "lr_monitor")


if __name__ == "__main__":
    CLI(CLIPDualEncoderModel, ImageRetrievalDataModule, save_config_callback=None)

from pytorch_lightning.utilities import cli

from dataloaders.data_module import ImageRetrievalDataModule
from models.clip_model import CLIPDualEncoderModel
from callbacks import LogPredictionCallback


class CLI(cli.LightningCLI):
    def add_arguments_to_parser(self, parser: cli.LightningArgumentParser) -> None:
        parser.link_arguments("model.text_encoder_alias", "data.tokenizer_alias")
        parser.add_lightning_class_args(
            LogPredictionCallback, "log_prediction_callback"
        )
        parser.link_arguments(
            "model.text_encoder_alias", "log_prediction_callback.tokenizer"
        )


if __name__ == "__main__":
    cli = CLI(CLIPDualEncoderModel, ImageRetrievalDataModule, save_config_callback=None, run=False)
    trainer = cli.trainer
    data_module = cli.datamodule
    

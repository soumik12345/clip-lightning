from pytorch_lightning.utilities import cli

from dataloaders.data_module import ImageRetrievalDataModule
from models.clip_model import CLIPDualEncoderModel


class CLI(cli.LightningCLI):
    def add_arguments_to_parser(self, parser: cli.LightningArgumentParser) -> None:
        parser.link_arguments("model.text_encoder_alias", "data.tokenizer_alias")


if __name__ == "__main__":
    CLI(CLIPDualEncoderModel, ImageRetrievalDataModule, save_config_callback=None)

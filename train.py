from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from TForge import TForgeDataLoader, LitTForge

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="proj_tforge")
    # wandb_logger = WandbLogger(project="proj_dummy")

    # model
    hyperparameter = {
        "input_vocab_size": 10000,
        "output_vocab_size": 10000,
        "translate_vocab_size": 10000,
        "max_positions": 256,
        "num_layers": 6,
        "num_heads": 8,
        "embed_dim": 512,
        "hidden_dim": 1024,
        "soft_align_dim": 512,
        "dropout_prob": 0.1,
        "lambda_threshold": 0.6,
    }

    lit_tforge = LitTForge(**hyperparameter)

    # dataloader
    tforge_dataloader = TForgeDataLoader(
        'tokenizer/v1/en_tokenizer_src', 'tokenizer/v1/en_tokenizer_tgt', 'tokenizer/v1/fr_tokenizer_tgt',
        40, 15, 80, 80
    )
    [train_dataloader, valid_dataloader, test_dataloader] = tforge_dataloader.get_dataloader(batch_size=64, types=["train", "valid", "test"])

    # train model
    trainer = pl.Trainer(max_epochs=140, logger=wandb_logger, devices=2, accelerator="gpu", strategy="ddp")
    trainer.fit(model=lit_tforge, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    # # save model & tokenizer
    lit_tforge.export_model('TForge_model/v2')

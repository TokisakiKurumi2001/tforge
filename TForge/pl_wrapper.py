import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List
from TForge import TForgeModel, TForgeConfig, TForgeScheduler

class LitTForge(pl.LightningModule):
    def __init__(
        self, input_vocab_size: int, output_vocab_size: int, translate_vocab_size: int,
        max_positions: int, num_layers: int, num_heads: int, embed_dim: int, hidden_dim: int,
        soft_align_dim: int, dropout_prob: float, lambda_threshold: float):
        super(LitTForge, self).__init__()
        self.lambda_threshold = lambda_threshold
        config = TForgeConfig(
            input_vocab_size=input_vocab_size,
            output_vocab_size=output_vocab_size,
            translate_vocab_size=translate_vocab_size,
            max_positions=max_positions,
            num_layers=num_layers, 
            num_heads=num_heads,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            soft_align_dim=soft_align_dim,
            dropout_prob=dropout_prob,
            load_pretrained=False
        )
        self.tforge = TForgeModel(config)
        self.reconstruct_loss = nn.NLLLoss(ignore_index=1)
        self.translate_loss = nn.NLLLoss(ignore_index=1)
        self.save_hyperparameters()

    def export_model(self, path):
        self.tforge.save_pretrained(path)

    def training_step(self, batch, batch_idx):
        tgt = batch.pop('decoder_tgt_labels')
        tsl = batch.pop('decoder_tsl_labels')
        out_tgt, out_tsl = self.tforge(batch)

        vocab_size_tgt = out_tgt.shape[-1]
        logits_tgt = out_tgt.reshape(-1, vocab_size_tgt)
        labels_tgt = tgt.reshape(-1).long()
        loss_tgt = self.reconstruct_loss(logits_tgt, labels_tgt)

        vocab_size_tsl = out_tsl.shape[-1]
        logits_tsl = out_tsl.reshape(-1, vocab_size_tsl)
        labels_tsl = tsl.reshape(-1).long()
        loss_tsl = self.translate_loss(logits_tsl, labels_tsl)

        loss = self.lambda_threshold * loss_tgt + (1 - self.lambda_threshold) * loss_tsl
        self.log("train/loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tgt = batch.pop('decoder_tgt_labels')
        tsl = batch.pop('decoder_tsl_labels')
        out_tgt, out_tsl = self.tforge(batch)

        vocab_size_tgt = out_tgt.shape[-1]
        logits_tgt = out_tgt.reshape(-1, vocab_size_tgt)
        labels_tgt = tgt.reshape(-1).long()
        loss_tgt = self.reconstruct_loss(logits_tgt, labels_tgt)

        vocab_size_tsl = out_tsl.shape[-1]
        logits_tsl = out_tsl.reshape(-1, vocab_size_tsl)
        labels_tsl = tsl.reshape(-1).long()
        loss_tsl = self.translate_loss(logits_tsl, labels_tsl)

        loss = self.lambda_threshold * loss_tgt + (1 - self.lambda_threshold) * loss_tsl
        self.log("valid/loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1.0e-9, lr=5e-5)
        scheduler = TForgeScheduler(optimizer)
        return [optimizer], [scheduler]
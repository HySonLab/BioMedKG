from typing import Any

import torch
from lightning import LightningModule

from torch_geometric.nn import DeepGraphInfomax
from transformers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from biomedkg.modules import GCNEncoder, AttentionFusion

class DGIModule(LightningModule):
    def __init__(self,
                 in_dim : int,
                 hidden_dim : int,
                 out_dim : int,
                 num_hidden_layers : int,
                 scheduler_type : str = "cosine",
                 learning_rate: float = 2e-4,
                 warm_up_ratio: float = 0.03,
                 ):
        super().__init__()
        assert scheduler_type in ["linear", "cosine"], "Only support 'cosine' and 'linear'"

        self.save_hyperparameters()

        self.modality_fuser = AttentionFusion(
            embed_dim=in_dim,
            norm=True,
            aggr="mean",
        )

        self.model = DeepGraphInfomax(
            hidden_channels=hidden_dim,
            encoder=GCNEncoder(
                in_dim=in_dim, 
                hidden_dim=hidden_dim, 
                out_dim=out_dim, 
                num_hidden_layers=num_hidden_layers
                ),
            summary=lambda z, *args, **kwargs: z.mean(dim=0).sigmoid(),
            corruption=self.corruption,
        )
        
        self.lr = learning_rate
        self.scheduler_type = scheduler_type
        self.warm_up_ratio = warm_up_ratio
    
    def forward(self, x, edge_index):
        x = self.modality_fuser(x)
        z = self.model.encoder(x, edge_index)
        return z
    
    def training_step(self, batch):
        x = self.modality_fuser(batch.x)
        pos_z, neg_z, summary = self.model(x, batch.edge_index)
        loss = self.model.loss(pos_z, neg_z, summary)

        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.modality_fuser(batch.x)
        pos_z, neg_z, summary = self.model(x, batch.edge_index)
        loss = self.model.loss(pos_z, neg_z, summary)

        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = self.modality_fuser(batch.x)
        pos_z, neg_z, summary = self.model(x, batch.edge_index)
        loss = self.model.loss(pos_z, neg_z, summary)

        self.log("test_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss
    
    @staticmethod
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0), device=x.device)], edge_index
    
    def configure_optimizers(self,):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        scheduler = self._get_scheduler(optimizer=optimizer)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def _get_scheduler(self, optimizer):
        scheduler_args = {
            "optimizer": optimizer,
            "num_training_steps": int(self.trainer.estimated_stepping_batches),
            "num_warmup_steps": int(self.trainer.estimated_stepping_batches * self.warm_up_ratio),
        }
        if self.scheduler_type == "linear":
            return get_linear_schedule_with_warmup(**scheduler_args)
        if self.scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(**scheduler_args)
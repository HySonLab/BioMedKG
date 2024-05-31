from typing import Callable
from collections import OrderedDict
import os

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MetricCollection, AUROC, AveragePrecision, RetrievalMRR, RetrievalHitRate

from torch_geometric.nn import GAE
from torch_geometric.utils import negative_sampling
from transformers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from biomedkg.modules import DistMult, TransE, ComplEx
from biomedkg.modules.encoder import RGAT, RGCN

class BenchmarkModule(LightningModule):
    def __init__(self,
                 encoder_path: str,
                 decoder_name : str,
                 out_dim : int,
                 num_relation : int,
                 scheduler_type : str = "cosine",
                 learning_rate: float = 2e-4,
                 warm_up_ratio: float = 0.03,
                 ):
        super().__init__()
        assert decoder_name in ["transe", "dismult", "complex"], "Only support 'transe', 'dismult', 'complex'."
        assert scheduler_type in ["linear", "cosine"], "Only support 'cosine' and 'linear'"

        self.save_hyperparameters()

        # self.load_pretrained_encoder(encoder_path)

        # Scoring functions 

        if decoder_name == "transe":
            self.decoder = TransE(
                num_relations=num_relation,
                hidden_channels=out_dim,
            )
        elif decoder_name == "dismult":
            self.decoder = DistMult(
                num_relations=num_relation,
                hidden_channels=out_dim,
            )
        elif decoder_name == "complex":
            self.decoder = ComplEx(
                num_relations=num_relation,
                hidden_channels=out_dim
            )
        else:
            raise NotImplemented
        
        self.lr = learning_rate
        self.scheduler_type = scheduler_type
        self.warm_up_ratio = warm_up_ratio

        # Metrics 

        metrics = MetricCollection(
            [
                AUROC(task="binary"),
                AveragePrecision(task="binary"),
                RetrievalHitRate(top_k=1),
                RetrievalHitRate(top_k=3),
                RetrievalHitRate(top_k=10),
                RetrievalMRR
            ]
        )

        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        
    def forward(self, x, edge_index, edge_type):
        return self.encoder(x, edge_index, edge_type)
    
    def training_step(self, batch):
        z = self.model.encode(batch.x, batch.edge_index, batch.edge_type)

        neg_edge_index = negative_sampling(batch.edge_index)

        pos_pred = self.model.decode(z, batch.edge_index, batch.edge_type)
        neg_pred = self.model.decode(z, neg_edge_index, batch.edge_type)
        pred = torch.cat([pos_pred, neg_pred])

        gt = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])

        cross_entropy_loss = F.binary_cross_entropy_with_logits(pred, gt)
        reg_loss = z.pow(2).mean() + self.model.decoder.rel_emb.pow(2).mean()
        loss = cross_entropy_loss + 1e-2 * reg_loss

        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        z = self.model.encode(batch.x, batch.edge_index, batch.edge_type)

        neg_edge_index = negative_sampling(batch.edge_index)

        pos_pred = self.model.decode(z, batch.edge_index, batch.edge_type)
        neg_pred = self.model.decode(z, neg_edge_index, batch.edge_type)
        pred = torch.cat([pos_pred, neg_pred])

        gt = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])

        self.valid_metrics.update(pred, gt.to(torch.int32))

        cross_entropy_loss = F.binary_cross_entropy_with_logits(pred, gt)
        reg_loss = z.pow(2).mean() + self.model.decoder.rel_emb.pow(2).mean()
        loss = cross_entropy_loss + 1e-2 * reg_loss

        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self,):
        output = self.valid_metrics.compute()
        self.log_dict(output)
        self.valid_metrics.reset()
    
    def test_step(self, batch,):
        z = self.model.encode(batch.x, batch.edge_index, batch.edge_type)

        neg_edge_index = negative_sampling(batch.edge_index)

        pos_pred = self.model.decode(z, batch.edge_index, batch.edge_type)
        neg_pred = self.model.decode(z, neg_edge_index, batch.edge_type)
        pred = torch.cat([pos_pred, neg_pred])

        gt = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])

        self.test_metrics.update(pred, gt.to(torch.int32))

    def on_test_epoch_end(self,):
        output = self.test_metrics.compute()
        self.log_dict(output)
        self.test_metrics.reset()
    
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


if __name__ == "__main__":
    encoder_path = "ckpt/kge/rgcn_dismult_1716172122"
    decoder_name = "complex"  # Choose from "transe", "dismult", "complex"
    out_dim = 128
    num_relation = 10
    scheduler_type = "cosine"
    learning_rate = 2e-4
    warm_up_ratio = 0.03

    # Create the BenchmarkModule instance
    benchmark_module = BenchmarkModule(
        encoder_path=encoder_path,
        decoder_name=decoder_name,
        out_dim=out_dim,
        num_relation=num_relation,
        scheduler_type=scheduler_type,
        learning_rate=learning_rate,
        warm_up_ratio=warm_up_ratio
    )

    # Print the encoder to check if it has been loaded correctly
    print("Loaded Encoder:")
    print(benchmark_module.encoder)
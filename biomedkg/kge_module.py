from typing import Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch_geometric.utils import negative_sampling
from torchmetrics import AUROC, AveragePrecision, F1Score, MetricCollection
from torchmetrics.wrappers import BootStrapper
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from biomedkg.factory import FusionFactory, KGEModelFactory
from biomedkg.utils.metrics import EdgeWisePrecision


class KGEModule(LightningModule):
    def __init__(
        self,
        encoder_name: str,
        decoder_name: str,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_hidden_layers: int,
        num_relation: int,
        num_heads: int,
        scheduler_type: str,
        learning_rate: float,
        warm_up_ratio: float,
        fuse_method: str,
        neg_ratio: int,
        node_init_method: str,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.modality_transform = (
            FusionFactory.create_fuser(method=fuse_method, embed_dim=in_dim)
            if node_init_method == "lm"
            else None
        )

        self.model = KGEModelFactory.get_model(
            encoder_name=encoder_name,
            decoder_name=decoder_name,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_hidden_layers=num_hidden_layers,
            num_relation=num_relation,
            num_heads=num_heads,
        )

        self.lr, self.scheduler_type, self.warm_up_ratio, self.neg_ratio = (
            learning_rate,
            scheduler_type,
            warm_up_ratio,
            neg_ratio,
        )

        metrics = MetricCollection(
            {
                "AUROC": BootStrapper(
                    AUROC(task="binary"),
                ),
                "AveragePrecision": BootStrapper(AveragePrecision(task="binary")),
                "F1": BootStrapper(F1Score(task="binary")),
            }
        )
        self._edge_index_map = dict()
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self._fix_edge_id = None

    def fusion_fn(self, x) -> torch.Tensor:
        if self.modality_transform:
            x = self.modality_transform(x)

        elif x.dim() == 3:
            x = torch.mean(x, dim=1)

        return x

    def sample_neg_edges(
        self, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sampling negative edges"""
        num_neg_samples = (
            self.neg_ratio * edge_index.size(-1) if self.neg_ratio else None
        )
        neg_edge_index = negative_sampling(edge_index, num_neg_samples=num_neg_samples)

        if self.neg_ratio:
            neg_edge_type = edge_type.repeat(self.neg_ratio)[
                torch.randperm(self.neg_ratio * edge_type.size(0))
            ]
        else:
            neg_edge_type = edge_type

        return neg_edge_index, neg_edge_type

    def forward(self, x, edge_index, edge_type):
        x = self.fusion_fn(x=x)

        return self.model.encode(x, edge_index, edge_type)

    def training_step(self, batch):
        x = self.fusion_fn(x=batch.x)

        if self._fix_edge_id is not None:
            batch.edge_type = torch.full_like(batch.edge_type, self._fix_edge_id)

        z = self.model.encode(x, batch.edge_index, batch.edge_type)

        neg_edge_index, neg_edge_type = self.sample_neg_edges(
            batch.edge_index, batch.edge_type
        )

        pos_pred = self.model.decode(z, batch.edge_index, batch.edge_type)
        neg_pred = self.model.decode(z, neg_edge_index, neg_edge_type)
        pred = torch.cat([pos_pred, neg_pred])

        gt = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])

        cross_entropy_loss = F.binary_cross_entropy_with_logits(pred, gt)
        reg_loss = z.pow(2).mean() + self.model.decoder.rel_emb.pow(2).mean()
        loss = cross_entropy_loss + 1e-2 * reg_loss

        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.fusion_fn(x=batch.x)

        if self._fix_edge_id is not None:
            batch.edge_type = torch.full_like(batch.edge_type, self._fix_edge_id)

        z = self.model.encode(x, batch.edge_index, batch.edge_type)

        neg_edge_index, neg_edge_type = self.sample_neg_edges(
            batch.edge_index, batch.edge_type
        )

        pos_pred = self.model.decode(z, batch.edge_index, batch.edge_type)
        neg_pred = self.model.decode(z, neg_edge_index, neg_edge_type)
        pred = torch.cat([pos_pred, neg_pred])

        gt = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])

        self.valid_metrics.update(pred, gt.to(torch.int32))
        if hasattr(self, "edge_wise_pre_valid"):
            self.edge_wise_pre_valid.update(pos_pred, batch.edge_type)

        cross_entropy_loss = F.binary_cross_entropy_with_logits(pred, gt)
        reg_loss = z.pow(2).mean() + self.model.decoder.rel_emb.pow(2).mean()
        loss = cross_entropy_loss + 1e-2 * reg_loss

        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        output = self.valid_metrics.compute()

        if hasattr(self, "edge_wise_pre_valid"):
            edge_wise_pre = self.edge_wise_pre_valid.compute()
            self.log_dict(edge_wise_pre)
            self.edge_wise_pre_valid.reset()

        self.log_dict(output)
        self.valid_metrics.reset()

    def test_step(self, batch, batch_idx):
        x = self.fusion_fn(x=batch.x)

        if self._fix_edge_id is not None:
            batch.edge_type = torch.full_like(batch.edge_type, self._fix_edge_id)

        z = self.model.encode(x, batch.edge_index, batch.edge_type)

        neg_edge_index, neg_edge_type = self.sample_neg_edges(
            batch.edge_index, batch.edge_type
        )

        pos_pred = self.model.decode(z, batch.edge_index, batch.edge_type)
        neg_pred = self.model.decode(z, neg_edge_index, neg_edge_type)
        pred = torch.cat([pos_pred, neg_pred])

        gt = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])

        self.test_metrics.update(pred, gt.to(torch.int32))
        if hasattr(self, "edge_wise_pre_test"):
            self.edge_wise_pre_test.update(pos_pred, batch.edge_type)

    def on_test_epoch_end(self):
        output = self.test_metrics.compute()

        if hasattr(self, "edge_wise_pre_test"):
            edge_wise_pre = self.edge_wise_pre_test.compute()
            self.log_dict(edge_wise_pre)
            self.edge_wise_pre_test.reset()

        self.log_dict(output)
        self.test_metrics.reset()
        return output

    def configure_optimizers(
        self,
    ):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = self._get_scheduler(optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def _get_scheduler(self, optimizer):
        scheduler_args = {
            "optimizer": optimizer,
            "num_training_steps": int(self.trainer.estimated_stepping_batches),
            "num_warmup_steps": int(
                self.trainer.estimated_stepping_batches * self.warm_up_ratio
            ),
        }
        if self.scheduler_type == "linear":
            return get_linear_schedule_with_warmup(**scheduler_args)
        if self.scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(**scheduler_args)

    @property
    def edge_mapping(self):
        return self._edge_index_map

    @edge_mapping.setter
    def edge_mapping(self, edge_mapping_dict: dict):
        self._edge_index_map = edge_mapping_dict
        self.edge_wise_pre_valid = EdgeWisePrecision(class_mapping=self._edge_index_map)
        self.edge_wise_pre_test = EdgeWisePrecision(class_mapping=self._edge_index_map)

    @property
    def fix_edge_id(self):
        return self._fix_edge_id

    @fix_edge_id.setter
    def fix_edge_id(self, edge_id: int):
        self._fix_edge_id = edge_id

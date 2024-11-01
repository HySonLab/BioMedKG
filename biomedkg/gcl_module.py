import GCL
import GCL.losses as L
import torch
import torch.nn.functional as F
from GCL.models import DualBranchContrast, SingleBranchContrast
from lightning import LightningModule
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from .factory import FusionFactory
from .model import GCNEncoder
from .model.gcl import DGI, GGD, GRACE


class BaseGCL(LightningModule):
    def __init__(
        self,
        model,
        embed_dim: int,
        scheduler_type: str,
        learning_rate: float,
        warm_up_ratio: float,
        feature_embedding_dim: int,
        contrast_model: GCL.models = None,
        fuse_method: str = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.modality_transform = FusionFactory.create_fuser(
            method=fuse_method, embed_dim=embed_dim
        )
        self.contrast_model = contrast_model
        self.lr = learning_rate
        self.scheduler_type = scheduler_type
        self.warm_up_ratio = warm_up_ratio
        self.feature_embedding_dim = feature_embedding_dim

    def fusion_fn(self, x):
        if self.modality_transform:
            x = self.modality_transform(x)

        elif x.dim() == 3:
            x = torch.mean(x, dim=1)

        return x

    def calculate_loss(self, x, edge_index):
        raise NotImplementedError

    def forward(self, x, edge_index):
        x = self.fusion_fn(x=x)
        z = self.model.encoder(x, edge_index)
        return z

    def training_step(self, batch):
        x = self.fusion_fn(x=batch.x)
        loss = self.calculate_loss(x, batch.edge_index)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.fusion_fn(x=batch.x)
        loss = self.calculate_loss(x, batch.edge_index)
        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = self.fusion_fn(x=batch.x)
        loss = self.calculate_loss(x, batch.edge_index)
        self.log("test_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

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


class DGIModule(BaseGCL):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_hidden_layers: int,
        scheduler_type: str = "cosine",
        learning_rate: float = 2e-4,
        warm_up_ratio: float = 0.03,
        fuse_method: str = None,
    ):
        self.save_hyperparameters(ignore=["model"])

        model = DGI(
            encoder=GCNEncoder(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                num_hidden_layers=num_hidden_layers,
            ),
            hidden_dim=hidden_dim,
        )

        contrast_model = SingleBranchContrast(loss=L.JSD(), mode="G2L")

        super().__init__(
            model=model,
            embed_dim=in_dim,
            scheduler_type=scheduler_type,
            learning_rate=learning_rate,
            warm_up_ratio=warm_up_ratio,
            feature_embedding_dim=in_dim,
            contrast_model=contrast_model,
            fuse_method=fuse_method,
        )

    def calculate_loss(self, x, edge_index):
        pos_z, summary, neg_z = self.model(x, edge_index)
        loss = self.contrast_model(h=pos_z, g=summary, hn=neg_z)
        return loss


class GRACEModule(BaseGCL):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_hidden_layers: int,
        scheduler_type: str = "cosine",
        learning_rate: float = 2e-4,
        warm_up_ratio: float = 0.03,
        fuse_method: str = None,
    ):
        self.save_hyperparameters(ignore=["model"])

        model = GRACE(
            encoder=GCNEncoder(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                num_hidden_layers=num_hidden_layers,
            ),
            hidden_dim=hidden_dim,
            proj_dim=hidden_dim,
        )

        contrast_model = DualBranchContrast(
            loss=L.InfoNCE(tau=0.2), mode="L2L", intraview_negs=True
        )

        super().__init__(
            model=model,
            embed_dim=in_dim,
            scheduler_type=scheduler_type,
            learning_rate=learning_rate,
            warm_up_ratio=warm_up_ratio,
            feature_embedding_dim=in_dim,
            contrast_model=contrast_model,
            fuse_method=fuse_method,
        )

    def calculate_loss(self, x, edge_index):
        _, z1, z2 = self.model(x, edge_index)
        h1, h2 = [self.model.project(x) for x in [z1, z2]]
        loss = self.contrast_model(h1, h2)
        return loss


class GGDModule(BaseGCL):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_hidden_layers: int,
        scheduler_type: str = "cosine",
        learning_rate: float = 2e-4,
        warm_up_ratio: float = 0.03,
        fuse_method: str = None,
    ):
        self.save_hyperparameters(ignore=["model"])

        model = GGD(
            encoder=GCNEncoder(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                num_hidden_layers=num_hidden_layers,
            ),
            hidden_dim=hidden_dim,
            n_proj=1,
            aug_p=0.5,
        )

        super().__init__(
            model=model,
            embed_dim=in_dim,
            scheduler_type=scheduler_type,
            learning_rate=learning_rate,
            warm_up_ratio=warm_up_ratio,
            feature_embedding_dim=in_dim,
            fuse_method=fuse_method,
        )

    def calculate_loss(self, x, edge_index):
        pos_h, neg_h = self.model(x, edge_index)
        pred = torch.cat([pos_h, neg_h])
        gt = torch.cat([torch.ones_like(pos_h), torch.zeros_like(neg_h)])
        loss = F.binary_cross_entropy_with_logits(pred, gt)
        return loss

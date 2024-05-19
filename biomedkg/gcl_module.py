import torch
import GCL.losses as L
from GCL.models import SingleBranchContrast, DualBranchContrast
import torch.nn.functional as F

from typing import Callable
from transformers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from biomedkg.modules.gcl import DGI, GRACE, GGD
from biomedkg.modules import GCNEncoder
from biomedkg.configs import node_settings
from biomedkg.modules.fusion import AttentionFusion, ReDAF

from lightning import LightningModule

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

        self.feature_embedding_dim = in_dim

        # Set up Modality fuser
        if node_settings.MODALITY_TRANSFORM_METHOD == "attention":
            self.modality_fuser = AttentionFusion(
                    embed_dim=node_settings.PRETRAINED_NODE_DIM,
                    norm=True,
                )
        elif node_settings.MODALITY_TRANSFORM_METHOD == "redaf":
            self.modality_fuser = ReDAF(
                embed_dim=node_settings.PRETRAINED_NODE_DIM,
                num_modalities = 2,
            )     
        else:
            self.modality_fuser = None

        self.modality_aggr = node_settings.MODALITY_MERGING_METHOD

        self.model = DGI(
            encoder=GCNEncoder(
                in_dim=in_dim, 
                hidden_dim=hidden_dim, 
                out_dim=out_dim, 
                num_hidden_layers=num_hidden_layers
                ),
            hidden_dim=hidden_dim,
        )

        self.contrast_model = SingleBranchContrast(loss=L.JSD(), mode="G2L")
        
        self.lr = learning_rate
        self.scheduler_type = scheduler_type
        self.warm_up_ratio = warm_up_ratio
    
    def forward(self, x, edge_index):

        # Reshape if does not apply fusion transformation
        if self.modality_fuser is None:
            x = x.view(x.size(0), -1, self.feature_embedding_dim)
            x = F.normalize(x, dim=-1)
        else:
            x = self.modality_fuser(x)

        if self.modality_aggr == "mean":
            x = torch.mean(x, dim=1)
        elif self.modality_aggr == "sum":
            x = torch.sum(x, dim=1)
        else:
            x = x.view(x.size(0), -1)      

        z = self.model.encoder(x, edge_index)

        return z
    
    def training_step(self, batch):

        # Reshape if does not apply transformation
        if self.modality_fuser is None:
            x = batch.x.view(batch.x.size(0), -1, self.feature_embedding_dim)
            x = F.normalize(x, dim=-1)
        else:
            x = self.modality_fuser(batch.x)

        # Modalities fusion
        if self.modality_aggr == "mean":
            x = torch.mean(x, dim=1)
        elif self.modality_aggr == "sum":
            x = torch.sum(x, dim=1)
        else:
            x = x.view(x.size(0), -1)

        # Contrastive Learning on Graph
        pos_z, summary, neg_z = self.model(x, batch.edge_index)
        loss = self.contrast_model(h=pos_z, g=summary, hn=neg_z)

        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        # Reshape if does not apply transformation
        if self.modality_fuser is None:
            x = batch.x.view(batch.x.size(0), -1, self.feature_embedding_dim)
            x = F.normalize(x, dim=-1)
        else:
            x = self.modality_fuser(batch.x)

        # Modalities fusion
        if self.modality_aggr == "mean":
            x = torch.mean(x, dim=1)
        elif self.modality_aggr == "sum":
            x = torch.sum(x, dim=1)
        else:
            x = x.view(x.size(0), -1)

        # Contrastive Learning on Graph
        pos_z, summary, neg_z = self.model(x, batch.edge_index)
        loss = self.contrast_model(h=pos_z, g=summary, hn=neg_z)

        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):

        # Reshape if does not apply transformation
        if self.modality_fuser is None:
            x = batch.x.view(batch.x.size(0), -1, self.feature_embedding_dim)
            x = F.normalize(x, dim=-1)
        else:
            x = self.modality_fuser(batch.x)

        # Modalities fusion
        if self.modality_aggr == "mean":
            x = torch.mean(x, dim=1)
        elif self.modality_aggr == "sum":
            x = torch.sum(x, dim=1)
        else:
            x = x.view(x.size(0), -1)

        # Contrastive Learning on Graph
        pos_z, summary, neg_z = self.model(x, batch.edge_index)
        loss = self.contrast_model(h=pos_z, g=summary, hn=neg_z)

        self.log("test_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss
    
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
    

class GRACEModule(LightningModule):
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

        self.feature_embedding_dim = in_dim
        
        # Set up Modality fuser
        if node_settings.MODALITY_TRANSFORM_METHOD == "attention":
            self.modality_fuser = AttentionFusion(
                    embed_dim=node_settings.PRETRAINED_NODE_DIM,
                    norm=True,
                )
        elif node_settings.MODALITY_TRANSFORM_METHOD == "redaf":
            self.modality_fuser = ReDAF(
                embed_dim=node_settings.PRETRAINED_NODE_DIM,
                num_modalities = 2,
            )     
        else:
            self.modality_fuser = None

        self.modality_aggr = node_settings.MODALITY_MERGING_METHOD

        self.model = GRACE(
            encoder=GCNEncoder(
                in_dim=in_dim, 
                hidden_dim=hidden_dim, 
                out_dim=out_dim, 
                num_hidden_layers=num_hidden_layers
                ),
            hidden_dim=hidden_dim,
            proj_dim=hidden_dim,
        )

        self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True)
        
        self.lr = learning_rate
        self.scheduler_type = scheduler_type
        self.warm_up_ratio = warm_up_ratio
    
    def forward(self, x, edge_index):

        # Reshape if does not apply fusion transformation
        if self.modality_fuser is None:
            x = x.view(x.size(0), -1, self.feature_embedding_dim)
            x = F.normalize(x, dim=-1)
        else:
            x = self.modality_fuser(x)

        if self.modality_aggr == "mean":
            x = torch.mean(x, dim=1)
        elif self.modality_aggr == "sum":
            x = torch.sum(x, dim=1)
        else:
            x = x.view(x.size(0), -1)      

        z = self.model.encoder(x, edge_index)

        return z
    
    def training_step(self, batch):

        # Reshape if does not apply transformation
        if self.modality_fuser is None:
            x = batch.x.view(batch.x.size(0), -1, self.feature_embedding_dim)
            x = F.normalize(x, dim=-1)
        else:
            x = self.modality_fuser(batch.x)

        # Modalities fusion
        if self.modality_aggr == "mean":
            x = torch.mean(x, dim=1)
        elif self.modality_aggr == "sum":
            x = torch.sum(x, dim=1)
        else:
            x = x.view(x.size(0), -1)

        # Contrastive Learning on Graph
        _, z1, z2 = self.model(x, batch.edge_index)
        h1, h2 = [self.model.project(x) for x in [z1, z2]]
        loss = self.contrast_model(h1, h2)

        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        # Reshape if does not apply transformation
        if self.modality_fuser is None:
            x = batch.x.view(batch.x.size(0), -1, self.feature_embedding_dim)
            x = F.normalize(x, dim=-1)
        else:
            x = self.modality_fuser(batch.x)

        # Modalities fusion
        if self.modality_aggr == "mean":
            x = torch.mean(x, dim=1)
        elif self.modality_aggr == "sum":
            x = torch.sum(x, dim=1)
        else:
            x = x.view(x.size(0), -1)

        # Contrastive Learning on Graph
        _, z1, z2 = self.model(x, batch.edge_index)
        h1, h2 = [self.model.project(x) for x in [z1, z2]]
        loss = self.contrast_model(h1, h2)

        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):

        # Reshape if does not apply transformation
        if self.modality_fuser is None:
            x = batch.x.view(batch.x.size(0), -1, self.feature_embedding_dim)
            x = F.normalize(x, dim=-1)
        else:
            x = self.modality_fuser(batch.x)

        # Modalities fusion
        if self.modality_aggr == "mean":
            x = torch.mean(x, dim=1)
        elif self.modality_aggr == "sum":
            x = torch.sum(x, dim=1)
        else:
            x = x.view(x.size(0), -1)

        # Contrastive Learning on Graph
        _, z1, z2 = self.model(x, batch.edge_index)
        h1, h2 = [self.model.project(x) for x in [z1, z2]]
        loss = self.contrast_model(h1, h2)

        self.log("test_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss
    
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
        

class GGDModule(LightningModule):
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

        self.feature_embedding_dim = in_dim

        # Set up Modality fuser
        if node_settings.MODALITY_TRANSFORM_METHOD == "attention":
            self.modality_fuser = AttentionFusion(
                    embed_dim=node_settings.PRETRAINED_NODE_DIM,
                    norm=True,
                )
        elif node_settings.MODALITY_TRANSFORM_METHOD == "redaf":
            self.modality_fuser = ReDAF(
                embed_dim=node_settings.PRETRAINED_NODE_DIM,
                num_modalities = 2,
            )     
        else:
            self.modality_fuser = None

        self.modality_aggr = node_settings.MODALITY_MERGING_METHOD

        self.model = GGD(
            encoder=GCNEncoder(
                in_dim=in_dim, 
                hidden_dim=hidden_dim, 
                out_dim=out_dim, 
                num_hidden_layers=num_hidden_layers
                ),
            hidden_dim=hidden_dim,
            n_proj=1,
            aug_p=0.5,
        )
        
        self.lr = learning_rate
        self.scheduler_type = scheduler_type
        self.warm_up_ratio = warm_up_ratio
    
    def forward(self, x, edge_index):

        # Reshape if does not apply fusion transformation
        if self.modality_fuser is None:
            x = x.view(x.size(0), -1, self.feature_embedding_dim)
        else:
            x = self.modality_fuser(x)
            x = F.normalize(x, dim=-1)

        if self.modality_aggr == "mean":
            x = torch.mean(x, dim=1)
        elif self.modality_aggr == "sum":
            x = torch.sum(x, dim=1)
        else:
            x = x.view(x.size(0), -1)      

        z = self.model.encoder(x, edge_index)

        return z
    
    def training_step(self, batch):

        # Reshape if does not apply transformation
        if self.modality_fuser is None:
            x = batch.x.view(batch.x.size(0), -1, self.feature_embedding_dim)
        else:
            x = self.modality_fuser(batch.x)
            x = F.normalize(x, dim=-1)

        # Modalities fusion
        if self.modality_aggr == "mean":
            x = torch.mean(x, dim=1)
        elif self.modality_aggr == "sum":
            x = torch.sum(x, dim=1)
        else:
            x = x.view(x.size(0), -1)

        # Contrastive Learning on Graph
        pos_h, neg_h = self.model(x, batch.edge_index)
        pred = torch.cat([pos_h, neg_h])
        gt = torch.cat([torch.ones_like(pos_h), torch.zeros_like(neg_h)])
        loss = F.binary_cross_entropy_with_logits(pred, gt)

        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        # Reshape if does not apply transformation
        if self.modality_fuser is None:
            x = batch.x.view(batch.x.size(0), -1, self.feature_embedding_dim)
        else:
            x = self.modality_fuser(batch.x)
            x = F.normalize(x, dim=-1)

        # Modalities fusion
        if self.modality_aggr == "mean":
            x = torch.mean(x, dim=1)
        elif self.modality_aggr == "sum":
            x = torch.sum(x, dim=1)
        else:
            x = x.view(x.size(0), -1)

        # Contrastive Learning on Graph
        pos_h, neg_h = self.model(x, batch.edge_index)
        pred = torch.cat([pos_h, neg_h])
        gt = torch.cat([torch.ones_like(pos_h), torch.zeros_like(neg_h)])
        loss = F.binary_cross_entropy_with_logits(pred, gt)

        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):

        # Reshape if does not apply transformation
        if self.modality_fuser is None:
            x = batch.x.view(batch.x.size(0), -1, self.feature_embedding_dim)
        else:
            x = self.modality_fuser(batch.x)
            x = F.normalize(x, dim=-1)

        # Modalities fusion
        if self.modality_aggr == "mean":
            x = torch.mean(x, dim=1)
        elif self.modality_aggr == "sum":
            x = torch.sum(x, dim=1)
        else:
            x = x.view(x.size(0), -1)

        # Contrastive Learning on Graph
        pos_h, neg_h = self.model(x, batch.edge_index)
        pred = torch.cat([pos_h, neg_h])
        gt = torch.cat([torch.ones_like(pos_h), torch.zeros_like(neg_h)])
        loss = F.binary_cross_entropy_with_logits(pred, gt)

        self.log("test_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss
    
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
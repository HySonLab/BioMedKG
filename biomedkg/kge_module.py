import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MetricCollection, AUROC, AveragePrecision

from torch_geometric.nn import GAE
from torch_geometric.utils import negative_sampling
from transformers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from biomedkg.modules import RGCN, RGAT, DistMult, TransE
from biomedkg.modules.fusion import AttentionFusion, ReDAF
from biomedkg.configs import kge_settings, node_settings

class KGEModule(LightningModule):
    def __init__(self,
                 encoder_name : str,
                 decoder_name : str,
                 in_dim : int,
                 hidden_dim : int,
                 out_dim : int,
                 num_hidden_layers : int,
                 num_relation : int,
                 num_heads : int,
                 scheduler_type : str = "cosine",
                 learning_rate: float = 2e-4,
                 warm_up_ratio: float = 0.03,
                 ):
        super().__init__()
        assert encoder_name in ["rgcn", "rgat"], "Only support 'rgcn' and 'rgat'."
        assert decoder_name in ["transe", "dismult"], "Only support 'transe' and 'dismult'."
        assert scheduler_type in ["linear", "cosine"], "Only support 'cosine' and 'linear'"

        self.save_hyperparameters()

        self.feature_embedding_dim = in_dim

        self.llm_init_node = False
        
        if kge_settings.KGE_NODE_INIT_METHOD == "llm":
            self.llm_init_node = True

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

            self.save_hyperparameters(
                {
                    "modality_fuser": node_settings.MODALITY_TRANSFORM_METHOD,
                    "modality_aggr": node_settings.MODALITY_MERGING_METHOD
                }
            )
        
        self.save_hyperparameters(
                {
                    "node_init_method": kge_settings.KGE_NODE_INIT_METHOD,
                }
            )

        if encoder_name == "rgcn":
            self.encoder = RGCN(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                num_hidden_layers=num_hidden_layers,
                num_relations=num_relation
            )
        
        elif encoder_name == "rgat":
            self.encoder = RGAT(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                num_hidden_layers=num_hidden_layers,
                num_heads=num_heads,
                num_relations=num_relation
            )
        else:
            raise NotImplemented
        
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
        else:
            raise NotImplemented

        self.model = GAE(
            encoder=self.encoder,
            decoder=self.decoder
        )
        
        self.lr = learning_rate
        self.scheduler_type = scheduler_type
        self.warm_up_ratio = warm_up_ratio

        metrics = MetricCollection(
            [
                AUROC(task="binary"),
                AveragePrecision(task="binary"),
            ]
        )
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
    
    def forward(self, x, edge_index, edge_type):
        if self.llm_init_node:
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

        return self.encoder(x, edge_index, edge_type)
    
    def training_step(self, batch):
        if self.llm_init_node:
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
        else:
            x = batch.x

        z = self.model.encode(x, batch.edge_index, batch.edge_type)

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

    def validation_step(self, batch, batch_idx):
        if self.llm_init_node:
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
        else:
            x = batch.x

        z = self.model.encode(x, batch.edge_index, batch.edge_type)

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
    
    def test_step(self, batch, batch_idx):
        if self.llm_init_node:
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
        else:
            x = batch.x

        z = self.model.encode(x, batch.edge_index, batch.edge_type)

        neg_edge_index = negative_sampling(batch.edge_index)

        pos_pred = self.model.decode(z, batch.edge_index, batch.edge_type)
        neg_pred = self.model.decode(z, neg_edge_index, batch.edge_type)
        pred = torch.cat([pos_pred, neg_pred])

        gt = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])

        self.test_metrics.update(pred, gt.to(torch.int32))

    def on_test_epoch_end(self, batch, batch_idx):
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
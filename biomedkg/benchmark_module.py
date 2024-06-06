from typing import Callable
from collections import OrderedDict

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MetricCollection, AUROC, AveragePrecision
from torchmetrics.retrieval import  RetrievalMRR, RetrievalHitRate

from torch_geometric.nn import GAE
from torch_geometric.utils import negative_sampling
from transformers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from biomedkg.configs import kge_settings
from biomedkg.modules import DistMult, TransE, ComplEx
from biomedkg.modules.encoder import RGAT, RGCN

class BenchmarkModule(LightningModule):
    def __init__(self,
                model_path: str,
                out_dim : int,
                in_dim : int,
                hidden_dim : int,
                num_hidden_layers : int,
                num_relation : int,
                scheduler_type : str = "cosine",
                learning_rate: float = 2e-4,
                warm_up_ratio: float = 0.03,
                 ):
        super().__init__()
        # assert decoder_name in ["transe", "dismult", "complex"], "Only support 'transe', 'dismult', 'complex'."
        assert scheduler_type in ["linear", "cosine"], "Only support 'cosine' and 'linear'"

        self.save_hyperparameters()

        # Load model 

        encoder_name, decoder_name = model_path.split("/")[-1].split("_")[0], model_path.split("/")[-1].split("_")[1]

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
                num_relations=num_relation
            )
        else:
            raise NotImplemented

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
                hidden_channels=out_dim*2
            )
        else:
            raise NotImplemented
        
        self.load_pretrained_checkpoint(model_path)

        self.model = GAE(
            encoder=self.encoder,
            decoder=self.decoder
        )
        
        self.lr = learning_rate
        self.scheduler_type = scheduler_type
        self.warm_up_ratio = warm_up_ratio

        # Metrics 

        metrics = MetricCollection(
            [
                AUROC(task="binary"),
                AveragePrecision(task="binary"),
                # RetrievalHitRate(top_k=1),
                # RetrievalHitRate(top_k=3),
                RetrievalHitRate(top_k=10),
                RetrievalMRR()
            ]
        )

        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def load_pretrained_checkpoint(self, path: str) -> Callable:

        # Load checkpoint 
        
        checkpoint_path = path + '/last.ckpt'
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        updated_state_dict = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            # Split the key at the first instance of '.' and remove the prefix part
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part in ["encoder", "decoder", "modality_fusion"]:
                    new_key = '.'.join(parts[i+1:])  # Join parts after "encoder" or "decoder"
                    updated_state_dict[new_key] = value
                    break
                if i == len(parts) - 1:  # If no "encoder" or "decoder" found, use the original key
                    updated_state_dict[key] = value

        # model_state_dict = self.encoder.state_dict()
        # updated_state_dict = {k: v for k, v in updated_state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
        # model_state_dict.update(updated_state_dict)

        # checkpoint["state_dict"] = updated_state_dict 
        
        # Load the pre-trained encoder from the specified path.
        self.encoder.load_state_dict(updated_state_dict, strict=False)
        self.decoder.load_state_dict(updated_state_dict, strict=False)
        
        
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
    model_path = "ckpt/kge/rgcn_dismult_1716172122"
    out_dim = 128
    num_relation = 8
    scheduler_type = "cosine"
    learning_rate = 2e-4
    warm_up_ratio = 0.03

    # Create the BenchmarkModule instance
    benchmark_module = BenchmarkModule(
        model_path=model_path,
        out_dim=out_dim,
        in_dim=128,
        hidden_dim=128,
        num_hidden_layers=2,
        num_relation=num_relation,
        scheduler_type=scheduler_type,
        learning_rate=learning_rate,
        warm_up_ratio=warm_up_ratio
    )

    # Print the encoder to check if it has been loaded correctly
    print("Loaded Encoder:")
    print(benchmark_module.encoder)

    print("Loaded Decoder:")
    print(benchmark_module.decoder)


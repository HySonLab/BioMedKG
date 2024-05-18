import os
import time
import json
import argparse

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from biomedkg.gcl_module import DGIModule, GRACEModule, GGDModule
from biomedkg.data_module import PrimeKGModule
from biomedkg.modules.node import EncodeNodeWithModality
from biomedkg.modules.utils import find_comet_api_key
from biomedkg.modules.fusion import AttentionFusion, ReDAF
from biomedkg.configs import train_settings, gcl_settings, data_settings, node_settings


def parse_opt():
        parser = argparse.ArgumentParser()
        parser.add_argument(
             '--task', 
             type=str, 
             action='store', 
             choices=['train', 'test'], 
             default='train', 
             help="Do training or testing task")
        
        parser.add_argument(
             '--model_name', 
             type=str, 
             action='store', 
             choices=['dgi', 'grace', 'ggd'], 
             default='dgi', 
             help="Select contrastive model name")
        
        parser.add_argument(
             '--node_type', 
             type=str, 
             action='store', 
             required=True,
             choices=['gene', 'drug', 'disease'], 
             help="Train contrastive learning on which node type")
        
        parser.add_argument(
             '--modality_transform', 
             type=str, 
             action='store', 
             default=None,
             choices=['attention', 'redaf', None], 
             help="Fusion module to apply on modalities embedding")

        parser.add_argument(
             '--modality_merging', 
             type=str, 
             action='store', 
             default='mean',
             choices=['mean', 'sum', 'concat'], 
             help="Modalities merging function")
        
        
        parser.add_argument(
             '--resume', 
             type=str, 
             default=None,
             required=False,
             help="Resume traning from ckpt")
        opt = parser.parse_args()
        return opt


def main(
          task:str, 
          model_name:str,
          node_type:str, 
          modality_transform: str,
          modality_merging: str,
          resume:str = None):
    print("\033[95m" + f"Graph Contrastive Learning on {node_type}" + "\033[0m")

    seed_everything(train_settings.SEED)

    if node_type == "gene":
        process_node = ['gene/protein']
    else:
        process_node = [node_type]

    data_module = PrimeKGModule(
        data_dir=data_settings.DATA_DIR,
        process_node_lst=process_node,
        process_edge_lst=data_settings.EDGES_LST,
        batch_size=train_settings.BATCH_SIZE,
        val_ratio=train_settings.VAL_RATIO,
        test_ratio=train_settings.TEST_RATIO,
        encoder=1(
            entity_type=node_type, 
            embed_path=os.path.join(os.path.dirname(data_settings.DATA_DIR), "embed"),
            )
    )

    data_module.setup(stage="split", embed_dim=node_settings.PRETRAINED_NODE_DIM)

    if modality_transform == "attention":
        modality_fuser = AttentionFusion(
                embed_dim=node_settings.PRETRAINED_NODE_DIM,
                norm=True,
            )
    elif modality_transform == "redaf":
        modality_fuser = ReDAF(
            embed_dim=node_settings.PRETRAINED_NODE_DIM,
            num_modalities = 2,
        )     
    else:
        modality_fuser = None
    
    if model_name == "dgi":
        model = DGIModule(
            in_dim=node_settings.PRETRAINED_NODE_DIM,
            hidden_dim=gcl_settings.GCL_HIDDEN_DIM,
            out_dim=node_settings.GCL_TRAINED_NODE_DIM,
            num_hidden_layers=gcl_settings.GCL_NUM_HIDDEN,
            modality_fuser=modality_fuser,
            modality_aggr=modality_merging,
            scheduler_type=train_settings.SCHEDULER_TYPE,
            learning_rate=train_settings.LEARNING_RATE,
            warm_up_ratio=train_settings.WARM_UP_RATIO,
        )
    elif model_name == "grace":
        model = GRACEModule(
            in_dim=node_settings.PRETRAINED_NODE_DIM,
            hidden_dim=gcl_settings.GCL_HIDDEN_DIM,
            out_dim=node_settings.GCL_TRAINED_NODE_DIM,
            num_hidden_layers=gcl_settings.GCL_NUM_HIDDEN,
            modality_fuser=modality_fuser,
            modality_aggr=modality_merging,
            scheduler_type=train_settings.SCHEDULER_TYPE,
            learning_rate=train_settings.LEARNING_RATE,
            warm_up_ratio=train_settings.WARM_UP_RATIO,
        )
    elif model_name == "ggd":
        model = GGDModule(
            in_dim=node_settings.PRETRAINED_NODE_DIM,
            hidden_dim=gcl_settings.GCL_HIDDEN_DIM,
            out_dim=node_settings.GCL_TRAINED_NODE_DIM,
            num_hidden_layers=gcl_settings.GCL_NUM_HIDDEN,
            modality_fuser=modality_fuser,
            modality_aggr=modality_merging,
            scheduler_type=train_settings.SCHEDULER_TYPE,
            learning_rate=train_settings.LEARNING_RATE,
            warm_up_ratio=train_settings.WARM_UP_RATIO,
        )
    else:
        raise NotImplementedError
    
    exp_name = str(int(time.time()))
    ckpt_path = os.path.join(train_settings.OUT_DIR, "gcl", node_type, f"{model_name}_{node_type}_{modality_transform}_{modality_merging}_{exp_name}")
    log_dir = os.path.join(train_settings.LOG_DIR, "gcl", node_type, f"{model_name}_{node_type}_{modality_transform}_{modality_merging}_{exp_name}")

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(os.path.join(ckpt_path, "node_mapping.json"), "w") as file:
        json.dump(data_module.primekg.mapping_dict, file, indent=4)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path, 
        monitor="val_loss", 
        save_top_k=3, 
        mode="min",
        save_last=True,
        )
    
    early_stopping = EarlyStopping(monitor="val_loss", mode="min")

    logger = CometLogger(
        api_key=find_comet_api_key(),
        project_name=f"BioMedKG-GCL-{node_type}",
        save_dir=log_dir,
        experiment_name=exp_name,
    )

    trainer_args = {
        "accelerator": "auto", 
        "log_every_n_steps": 10,
        "max_epochs": train_settings.EPOCHS,
        "check_val_every_n_epoch": train_settings.VAL_EVERY_N_EPOCH,
        "default_root_dir": ckpt_path,
        "enable_checkpointing": True, 
        "logger": logger, 
        "callbacks": [checkpoint_callback, early_stopping], 
        "deterministic": True, 
        "gradient_clip_val": 1.0,
    }

    if isinstance(train_settings.DEVICES, list) and len(train_settings.DEVICES) > 1:
        trainer_args.update(
            {
                "devices": train_settings.DEVICES,
                "strategy": "ddp"
            }
        )
    else:
        if torch.cuda.device_count() > 0:
            trainer_args.update(
                {
                    "devices": train_settings.DEVICES,
                }
            )

    trainer = Trainer(**trainer_args)
    
    if task == "train":
        trainer.fit(
            model=model,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader(),
            ckpt_path=resume 
        )
    elif task == "test":
        assert resume is not None, "Please specify checkpoint path."
        trainer.test(
             model=model,
             dataloaders=data_module.test_dataloader(),
             ckpt_path=resume,
        )


if __name__ == "__main__":
    main(**vars(parse_opt()))
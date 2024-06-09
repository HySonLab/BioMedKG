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
             '--ckpt_path', 
             type=str, 
             default=None,
             required=False,
             help="Path to checkpoint")
        opt = parser.parse_args()
        return opt


def main(
          task:str, 
          model_name:str,
          node_type:str, 
          ckpt_path:str = None):
    print("\033[95m" + f"Graph Contrastive Learning on {node_type}" + "\033[0m")

    seed_everything(train_settings.SEED)

    # Process data
    if node_type == "gene":
        process_node = ['gene/protein']
    else:
        process_node = [node_type]

    data_module = PrimeKGModule(
        process_node_lst=process_node,
        encoder=EncodeNodeWithModality(
            entity_type=node_type, 
            embed_path=os.path.join(os.path.dirname(data_settings.DATA_DIR), "embed"),
            )
    )

    data_module.setup(stage="split", embed_dim=node_settings.PRETRAINED_NODE_DIM)

    gcl_kwargs = {
        "in_dim": node_settings.PRETRAINED_NODE_DIM,
        "hidden_dim": gcl_settings.GCL_HIDDEN_DIM,
        "out_dim": node_settings.GCL_TRAINED_NODE_DIM,
        "num_hidden_layers": gcl_settings.GCL_NUM_HIDDEN,
        "scheduler_type": train_settings.SCHEDULER_TYPE,
        "learning_rate": train_settings.LEARNING_RATE,
        "warm_up_ratio": train_settings.WARM_UP_RATIO,
    }

    # Initialize GCL module
    if model_name == "dgi":
        model = DGIModule(**gcl_kwargs)
    elif model_name == "grace":
        model = GRACEModule(**gcl_kwargs)
    elif model_name == "ggd":
        model = GGDModule(**gcl_kwargs)
    else:
        raise NotImplementedError

    # Prepare trainer args
    trainer_args = {
        "accelerator": "auto", 
        "log_every_n_steps": 10,
        "deterministic": True, 
    }

    # Setup multiple GPUs training
    if isinstance(train_settings.DEVICES, list) and len(train_settings.DEVICES) > 1:
        trainer_args.update(
            {
                "devices": train_settings.DEVICES,
                "strategy": "ddp"
            }
        )
    else:
        if torch.cuda.device_count() > 1:
            trainer_args.update(
                {
                    "devices": train_settings.DEVICES,
                }
            )

    # Train
    if task == "train":
        exp_name = str(int(time.time()))
        ckpt_dir = os.path.join(train_settings.OUT_DIR, "gcl", node_type, f"{model_name}_{node_type}_{node_settings.MODALITY_TRANSFORM_METHOD}_{node_settings.MODALITY_MERGING_METHOD}_{exp_name}")
        log_dir = os.path.join(train_settings.LOG_DIR, "gcl", node_type, f"{model_name}_{node_type}_{node_settings.MODALITY_TRANSFORM_METHOD}_{node_settings.MODALITY_MERGING_METHOD}_{exp_name}")

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with open(os.path.join(ckpt_dir, "node_mapping.json"), "w") as file:
            json.dump(data_module.primekg.mapping_dict, file, indent=4)

        # Setup callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir, 
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

        trainer_args.update(
            {
                "max_epochs": train_settings.EPOCHS,
                "check_val_every_n_epoch": train_settings.VAL_EVERY_N_EPOCH,
                "enable_checkpointing": True,     
                "gradient_clip_val": 1.0,
                "callbacks": [checkpoint_callback, early_stopping],
                "default_root_dir": ckpt_dir,
                "logger": logger, 
            }
        )

        trainer = Trainer(**trainer_args)

        trainer.fit(
            model=model,
            train_dataloaders=data_module.train_dataloader(loader_type="neighbor"),
            val_dataloaders=data_module.val_dataloader(loader_type="neighbor"),
            ckpt_path=ckpt_path 
        )

    # Test
    elif task == "test":
        assert ckpt_path is not None, "Please specify checkpoint path."
        trainer = Trainer(**trainer_args)
        trainer.test(
             model=model,
             dataloaders=data_module.test_dataloader(loader_type="neighbor"),
             ckpt_path=ckpt_path,
        )
    
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main(**vars(parse_opt()))
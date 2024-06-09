import os
import time
import json
import argparse


import comet_ml
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from biomedkg.kge_module import KGEModule
from biomedkg.modules.node import EncodeNode, EncodeNodeWithModality
from biomedkg.data_module import PrimeKGModule, BioKGModule
from biomedkg.modules.utils import find_comet_api_key
from biomedkg.configs import train_settings, kge_settings, data_settings, node_settings

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
             '--ckpt_path', 
             type=str, 
             default=None,
             required=False,
             help="Path to checkpoint")
        
        parser.add_argument(
             '--gcl_embed_path',
             type=str,
             default=None,
             required=False,
             help="Path to your GCL embedding")

        parser.add_argument(
             '--run_benchmark',
             action="store_true",
             help="Path to your GCL embedding")
        
        opt = parser.parse_args()
        return opt

def main(
        task:str, 
        ckpt_path: str = None, 
        gcl_embed_path: str = None,
        run_benchmark: bool = False,
        ):
    seed_everything(train_settings.SEED)

    embed_dim = None
    node_init_method = kge_settings.KGE_NODE_INIT_METHOD

    if node_init_method == "gcl":
        assert gcl_embed_path is not None
        encoder = EncodeNode(
            embed_path=gcl_embed_path
            )
        embed_dim = node_settings.GCL_TRAINED_NODE_DIM
    elif node_init_method == "llm":
        encoder = EncodeNodeWithModality(
            entity_type=list(data_settings.NODES_LST), 
            embed_path=os.path.join(os.path.dirname(data_settings.DATA_DIR), "embed"),
            )
        embed_dim = node_settings.PRETRAINED_NODE_DIM
    elif node_init_method == "random":
        encoder = None
        embed_dim = node_settings.GCL_TRAINED_NODE_DIM
    else:
        raise NotImplementedError

    # Setup data module
    if run_benchmark:
        assert ckpt_path is not None

        model = KGEModule.load_from_checkpoint(ckpt_path)
        # In PrimeKG, drug - gene relation is one
        model.select_edge_type_id = 1
        data_module = BioKGModule(encoder=encoder)
        data_module.setup(stage="split", embed_dim=embed_dim)

    else:
        data_module = PrimeKGModule(encoder=encoder,)
        data_module.setup(stage="split", embed_dim=embed_dim)
        model = KGEModule(
            in_dim=embed_dim,
            num_relation=data_module.data.num_edge_types,
        )

    # Setup logging/ckpt path
    if ckpt_path is None:
        exp_name = str(int(time.time()))
        if gcl_embed_path is not None:
            model_name = gcl_embed_path.split('/')[-1]
            ckpt_dir = os.path.join(train_settings.OUT_DIR, "kge", f"{node_init_method}_{model_name}_{kge_settings.KGE_ENCODER}_{kge_settings.KGE_DECODER}_{exp_name}")
            log_dir = os.path.join(train_settings.LOG_DIR, "kge", f"{node_init_method}_{model_name}_{kge_settings.KGE_ENCODER}_{kge_settings.KGE_DECODER}_{exp_name}")
        else:   
            ckpt_dir = os.path.join(train_settings.OUT_DIR, "kge", f"{node_init_method}_{kge_settings.KGE_ENCODER}_{kge_settings.KGE_DECODER}_{exp_name}")
            log_dir = os.path.join(train_settings.LOG_DIR, "kge", f"{node_init_method}_{kge_settings.KGE_ENCODER}_{kge_settings.KGE_DECODER}_{exp_name}")

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with open(os.path.join(ckpt_dir, "node_mapping.json"), "w") as file:
            json.dump(data_module.primekg.mapping_dict, file, indent=4)
    else:
        if run_benchmark:
            ckpt_dir = "benchmark" + os.path.dirname(ckpt_path)
            exp_name = "benchmark" + str(os.path.basename(ckpt_dir).split("_")[-1])
        else:
            ckpt_dir = os.path.dirname(ckpt_path)
            exp_name = str(os.path.basename(ckpt_dir).split("_")[-1])

        log_dir = ckpt_dir.replace(os.path.basename(train_settings.OUT_DIR), os.path.basename(train_settings.LOG_DIR))

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir, 
        monitor="val_loss", 
        save_top_k=3, 
        mode="min",
        save_last=True,
        )
    
    early_stopping = EarlyStopping(monitor="val_BinaryAUROC", mode="max", min_delta=0.001, patience=1)

    logger = CometLogger(
        api_key=find_comet_api_key(),
        project_name="BioMedKG-KGE",
        save_dir=log_dir,
        experiment_name=exp_name,
    )

    # Prepare trainer args
    trainer_args = {
        "accelerator": "auto", 
        "log_every_n_steps": 10,
        "default_root_dir": ckpt_dir,
        "logger": logger, 
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
        trainer_args.update(
            {
                "max_epochs": train_settings.EPOCHS,
                "check_val_every_n_epoch": train_settings.VAL_EVERY_N_EPOCH,
                "enable_checkpointing": True,     
                "gradient_clip_val": 1.0,
                "callbacks": [checkpoint_callback, early_stopping],
            }
        )

        trainer = Trainer(**trainer_args)

        trainer.fit(
            model=model,
            train_dataloaders=data_module.train_dataloader(loader_type="graph_saint"),
            val_dataloaders=data_module.val_dataloader(loader_type="graph_saint"),
            ckpt_path=ckpt_path 
        )

    # Test
    elif task == "test":
        assert ckpt_path is not None, "Please specify checkpoint path."
        trainer = Trainer(**trainer_args)
        trainer.test(
             model=model,
             dataloaders=data_module.test_dataloader(loader_type="graph_saint"),
             ckpt_path=ckpt_path,
        )
    
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main(**vars(parse_opt()))
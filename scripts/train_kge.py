import os
import time
import json
import argparse


import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping


from biomedkg.kge_module import KGEModule
from biomedkg.modules.node import EncodeNode, EncodeNodeWithModality
from biomedkg.data_module import PrimeKGModule
from biomedkg.modules.utils import find_comet_api_key
from biomedkg.configs import train_settings, kge_settings, data_settings, node_settings


def parse_opt():
        parser = argparse.ArgumentParser()
        parser.add_argument(
             '--gcl_embed_path',
             type=str,
             default=None,
             required=False,
             help="Path to your GCL embedding")
       
        parser.add_argument(
             '--node_init_method',
             type=str,
             default=kge_settings.KGE_NODE_INIT_METHOD,
             required=False,
             help="Node init method to train KGE")
       
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
        opt = parser.parse_args()
        return opt




def main(task:str, ckpt_path:str = None, gcl_embed_path:str=None, node_init_method:str=None):
    seed_everything(train_settings.SEED)


    if node_init_method == "gcl":
        encoder = EncodeNode(
            embed_path=gcl_embed_path
            )
    elif node_init_method == "llm":
        encoder = EncodeNodeWithModality(
            entity_type=list(data_settings.NODES_LST),
            embed_path=os.path.join(os.path.dirname(data_settings.DATA_DIR), "embed"),
            )
    elif node_init_method == "random":
        encoder = None
    else:
        raise NotImplementedError
   
    if node_init_method == "gcl" or node_init_method == "random":
        embed_dim = node_settings.GCL_TRAINED_NODE_DIM
    elif node_init_method == "llm":
        embed_dim = node_settings.PRETRAINED_NODE_DIM
    else:
        embed_dim = None


    # Setup data module
    data_module = PrimeKGModule(
        data_dir=data_settings.DATA_DIR,
        process_node_lst=data_settings.NODES_LST,
        process_edge_lst=data_settings.EDGES_LST,
        batch_size=train_settings.BATCH_SIZE,
        val_ratio=train_settings.VAL_RATIO,
        test_ratio=train_settings.TEST_RATIO,
        encoder=encoder,
    )


    data_module.setup(stage="split", embed_dim=embed_dim)


    # Initialize KGE Module
    model = KGEModule(
        encoder_name=kge_settings.KGE_ENCODER,
        decoder_name=kge_settings.KGE_DECODER,
        in_dim=embed_dim,
        hidden_dim=kge_settings.KGE_HIDDEN_DIM,
        out_dim=node_settings.KGE_TRAINED_NODE_DIM,
        num_hidden_layers=kge_settings.KGE_NUM_HIDDEN,
        num_relation=data_module.data.num_edge_types,
        num_heads=kge_settings.KGE_NUM_HEAD,
        scheduler_type=train_settings.SCHEDULER_TYPE,
        learning_rate=train_settings.LEARNING_RATE,
        warm_up_ratio=train_settings.WARM_UP_RATIO,
    )


    # Setup logging/ckpt path
    if ckpt_path is None:
        exp_name = str(int(time.time()))
        ckpt_dir = os.path.join(train_settings.OUT_DIR, "kge", f"{kge_settings.KGE_ENCODER}_{kge_settings.KGE_DECODER}_{exp_name}")
        log_dir = os.path.join(train_settings.LOG_DIR, "kge", f"{kge_settings.KGE_ENCODER}_{kge_settings.KGE_DECODER}_{exp_name}")


        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
       
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)


        with open(os.path.join(ckpt_dir, "node_mapping.json"), "w") as file:
            json.dump(data_module.primekg.mapping_dict, file, indent=4)
    else:
        ckpt_dir = os.path.dirname(ckpt_path)
        log_dir = ckpt_dir.replace(os.path.basename(train_settings.OUT_DIR), os.path.basename(train_settings.LOG_DIR))
        exp_name = str(os.path.basename(ckpt_dir).split("_")[-1])


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
        if torch.cuda.device_count() > 0:
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
        trainer.test(
             model=model,
             dataloaders=data_module.test_dataloader(loader_type="graph_saint"),
             ckpt_path=ckpt_path,
        )
   
    else:
        raise NotImplementedError




if __name__ == "__main__":
    main(**vars(parse_opt()))

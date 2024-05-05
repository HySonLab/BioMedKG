import os
import time
import argparse

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from biomedkg.gcl_module import DGIModule
from biomedkg.data_module import PrimeKGModule
from biomedkg.modules.node import EncodeNodeWithModality
from biomedkg.modules.utils import find_comet_api_key
from biomedkg.configs import kge_train_settings, kge_settings, data_settings


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
             '--node_type', 
             type=str, 
             action='store', 
             required=True,
             choices=['gene', 'drug', 'disease'], 
             help="Train contrastive learning on which node type")
        
        parser.add_argument(
             '--ckpt', 
             type=str, 
             default=None,
             required=False,
             help="ckpt path")
        opt = parser.parse_args()
        return opt


def main(task:str, node_type:str, ckpt:str = None):
    print(f"Graph Contrastive Learning on {node_type}")

    seed_everything(kge_train_settings.SEED)

    if node_type == "gene":
        process_node = ['gene/protein']
    else:
        process_node = [node_type]

    data_module = PrimeKGModule(
        data_dir=data_settings.DATA_DIR,
        process_node_lst=process_node,
        process_edge_lst=data_settings.EDGES_LST,
        batch_size=kge_train_settings.BATCH_SIZE,
        val_ratio=kge_train_settings.VAL_RATIO,
        test_ratio=kge_train_settings.TEST_RATIO,
        num_steps=kge_train_settings.STEP_PER_EPOCH,
        encoder=EncodeNodeWithModality(entity_type=node_type, embed_path="./data/embed")
    )

    data_module.setup()

    model = DGIModule(
        in_dim=kge_settings.IN_DIMS,
        hidden_dim=kge_settings.HIDDEN_DIM,
        out_dim=kge_settings.OUT_DIM,
        num_hidden_layers=kge_settings.NUM_HIDDEN,
        scheduler_type=kge_train_settings.SCHEDULER_TYPE,
        learning_rate=kge_train_settings.LEARNING_RATE,
        warm_up_ratio=kge_train_settings.WARM_UP_RATIO,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=kge_train_settings.OUT_DIR, 
        monitor="val_loss", 
        save_top_k=3, 
        mode="min"
        )
    
    early_stopping = EarlyStopping(monitor="val_loss", mode="min")

    logger = CometLogger(
        api_key=find_comet_api_key(),
        project_name=f"BioMedKG-GCL-{node_type}",
        save_dir=kge_train_settings.LOG_DIR,
        experiment_name=str(int(time.time())),
    )

    trainer = Trainer(
        accelerator="auto", 
        log_every_n_steps=10,
        max_epochs=kge_train_settings.EPOCHS,
        check_val_every_n_epoch=kge_train_settings.VAL_EVERY_N_EPOCH,
        default_root_dir=kge_train_settings.OUT_DIR,
        enable_checkpointing=True, 
        logger=logger, 
        callbacks=[checkpoint_callback, early_stopping], 
        deterministic=True, 
        gradient_clip_val=1.0,
        )
    
    if task == "train":
        trainer.fit(
            model=model,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader(),
            ckpt_path=ckpt 
        )
    elif task == "test":
        assert ckpt is not None, "Please specify checkpoint path."
        trainer.test(
             model=model,
             dataloaders=data_module.test_dataloader,
             ckpt_path=ckpt,
        )


if __name__ == "__main__":
    main(**vars(parse_opt()))
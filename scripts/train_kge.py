import os
import time
import argparse

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from biomedkg.kge_module import KGEModule
from biomedkg.data_module import PrimeKGModule
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
             '--ckpt', 
             type=str, 
             default=None,
             required=False,
             help="ckpt path")
        opt = parser.parse_args()
        return opt


def main(task:str, ckpt:str = None):
    seed_everything(kge_train_settings.SEED)

    data_module = PrimeKGModule(
        data_dir=data_settings.DATA_DIR,
        process_node_lst=data_settings.NODES_LST,
        process_edge_lst=data_settings.EDGES_LST,
        batch_size=kge_train_settings.BATCH_SIZE,
        val_ratio=kge_train_settings.VAL_RATIO,
        test_ratio=kge_train_settings.TEST_RATIO,
        num_steps=kge_train_settings.STEP_PER_EPOCH,
    )

    data_module.setup()

    model = KGEModule(
        encoder_name=kge_settings.ENCODER_MODEL_NAME,
        decoder_name=kge_settings.DECODER_MODEL_NAME,
        in_dim=kge_settings.IN_DIMS,
        hidden_dim=kge_settings.HIDDEN_DIM,
        out_dim=kge_settings.OUT_DIM,
        num_hidden_layers=kge_settings.NUM_HIDDEN,
        num_relation=data_module.data.num_edge_types,
        num_heads=kge_settings.NUM_HEAD,
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
        project_name="BioMedKG KGE",
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
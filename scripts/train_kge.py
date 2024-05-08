import os
import time
import json
import argparse

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from biomedkg.kge_module import KGEModule
from biomedkg.data_module import PrimeKGModule
from biomedkg.modules.utils import find_comet_api_key
from biomedkg.configs import train_settings, kge_settings, data_settings


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
    seed_everything(train_settings.SEED)

    data_module = PrimeKGModule(
        data_dir=data_settings.DATA_DIR,
        process_node_lst=data_settings.NODES_LST,
        process_edge_lst=data_settings.EDGES_LST,
        batch_size=train_settings.BATCH_SIZE,
        val_ratio=train_settings.VAL_RATIO,
        test_ratio=train_settings.TEST_RATIO,
    )

    data_module.setup()

    model = KGEModule(
        encoder_name=kge_settings.KGE_ENCODER_MODEL_NAME,
        decoder_name=kge_settings.KGE_DECODER_MODEL_NAME,
        in_dim=kge_settings.KGE_IN_DIMS,
        hidden_dim=kge_settings.KGE_HIDDEN_DIM,
        out_dim=kge_settings.KGE_OUT_DIM,
        num_hidden_layers=kge_settings.KGE_NUM_HIDDEN,
        num_relation=data_module.data.num_edge_types,
        num_heads=kge_settings.KGE_NUM_HEAD,
        scheduler_type=train_settings.SCHEDULER_TYPE,
        learning_rate=train_settings.LEARNING_RATE,
        warm_up_ratio=train_settings.WARM_UP_RATIO,
    )

    ckpt_path = os.path.join(train_settings.OUT_DIR, "kge", f"{kge_settings.KGE_ENCODER_MODEL_NAME}_{kge_settings.KGE_DECODER_MODEL_NAME}_{int(time.time())}")

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    with open(os.path.join(ckpt_path, "node_mapping.json"), "w") as file:
        json.dump(data_module.primekg.mapping_dict, file, indent=4)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(train_settings.OUT_DIR, "kge"), 
        monitor="val_loss", 
        save_top_k=3, 
        mode="min"
        )
    
    early_stopping = EarlyStopping(monitor="val_loss", mode="min")

    logger = CometLogger(
        api_key=find_comet_api_key(),
        project_name="BioMedKG KGE",
        save_dir=train_settings.LOG_DIR,
        experiment_name=str(int(time.time())),
    )

    trainer = Trainer(
        accelerator="auto", 
        log_every_n_steps=10,
        max_epochs=train_settings.EPOCHS,
        check_val_every_n_epoch=train_settings.VAL_EVERY_N_EPOCH,
        default_root_dir=train_settings.OUT_DIR,
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
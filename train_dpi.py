import os
import time

import hydra
from hydra.utils import instantiate
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CometLogger
from omegaconf import DictConfig

from biomedkg.common import find_comet_api_key
from biomedkg.kge_module import KGEModule


@hydra.main(version_base=None, config_path="configs", config_name="dpi")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    data_module = instantiate(
        cfg.data, gcl_model=cfg.gcl_model, gcl_fuse_method=cfg.gcl_fuse_method
    )
    data_module.setup(stage="split")

    if not cfg.pretrained_path.endswith(".ckpt"):
        model = KGEModule(
            **cfg.model,
            num_relation=data_module.data.num_edge_types,
            neg_ratio=cfg.neg_ratio,
            node_init_method=cfg.data.node_init_method,
        )
    else:
        model = KGEModule.load_from_checkpoint(cfg.pretrained_path)
        model.fix_edge_id = (
            1  # Because in PrimeKG, 1 is the index of protein-drug relationship
        )
        model.neg_ratio = cfg.neg_ratio

    model.edge_mapping = data_module.edge_map_index

    exp_name = (
        f"{cfg.model.encoder_name}_{cfg.model.decoder_name}_{cfg.data.node_init_method}"
    )
    if cfg.data.node_init_method == "gcl":
        exp_name = exp_name + f"{cfg.gcl_model}_{cfg.gcl_fuse_method}"
    exp_name = exp_name + str(int(time.time()))
    ckpt_dir = os.path.join(cfg.ckpt_dir, "dpi", exp_name)
    log_dir = os.path.join(cfg.log_dir, "dpi", exp_name)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    trainer_args = {
        "accelerator": "auto",
        "log_every_n_steps": 10,
        "deterministic": True,
        "devices": cfg.devices,
    }

    if cfg.debug:
        trainer_args["fast_dev_run"] = True

    # Setup callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True,
    )

    logger = CometLogger(
        api_key=find_comet_api_key(),
        project_name="BioMedKG-DPI",
        save_dir=log_dir,
        experiment_name=exp_name,
    )

    trainer_args.update(
        {
            "max_epochs": cfg.epochs,
            "check_val_every_n_epoch": cfg.val_every_epoch,
            "enable_checkpointing": True,
            "gradient_clip_val": 1.0,
            "callbacks": [checkpoint_callback],
            "default_root_dir": ckpt_dir,
            "logger": logger,
        }
    )

    trainer = Trainer(**trainer_args)

    trainer.fit(
        model=model,
        train_dataloaders=data_module.train_dataloader(loader_type="saint"),
        val_dataloaders=data_module.val_dataloader(loader_type="saint"),
    )

    test_args = {
        "model": model,
        "dataloaders": data_module.test_dataloader(loader_type="saint"),
    }

    if not cfg.debug:
        test_args["ckpt_path"] = "best"

    trainer.test(**test_args)


if __name__ == "__main__":
    main()

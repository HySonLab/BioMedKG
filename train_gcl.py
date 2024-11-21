import os
import time

import hydra
from hydra.utils import instantiate
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CometLogger
from omegaconf import DictConfig

from biomedkg.common import find_comet_api_key
from biomedkg.gcl_module import DGIModule, GGDModule, GRACEModule


def create_gcl_model(cfg: DictConfig):
    model_name = cfg.model_name

    gcl_kwargs = {
        "in_dim": cfg.in_dim,
        "hidden_dim": cfg.hidden_dim,
        "out_dim": cfg.out_dim,
        "num_hidden_layers": cfg.num_hidden_layers,
        "scheduler_type": cfg.scheduler_type,
        "learning_rate": cfg.learning_rate,
        "warm_up_ratio": cfg.warm_up_ratio,
        "fuse_method": cfg.fuse_method,
    }

    if model_name == "dgi":
        model = DGIModule(**gcl_kwargs)
    elif model_name == "grace":
        model = GRACEModule(**gcl_kwargs)
    elif model_name == "ggd":
        model = GGDModule(**gcl_kwargs)
    else:
        raise NotImplementedError

    return model


@hydra.main(version_base=None, config_path="configs", config_name="gcl")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    log_name = f"{cfg.model.model_name}_{cfg.model.fuse_method}_{cfg.data.node_init_method}_{str(int(time.time()))}"
    ckpt_dir = os.path.join(cfg.ckpt_dir, "gcl", cfg.data.node_type, log_name)
    log_dir = os.path.join(cfg.log_dir, "gcl", cfg.data.node_type, log_name)

    if isinstance(cfg.data.node_type, list) and len(cfg.data.node_type) > 1:
        raise ValueError("Please select only one node type")

    if cfg.data.node_type.startswith("gene"):
        cfg.data.node_type = ["gene/protein"]
    else:
        cfg.data.node_type = [cfg.data.node_type]

    data_module = instantiate(cfg.data)
    data_module.setup(stage="split")

    model = create_gcl_model(cfg=cfg.model)

    # Prepare trainer args
    trainer_args = {
        "accelerator": "auto",
        "log_every_n_steps": 10,
        "deterministic": True,
        "devices": cfg.devices,
    }

    # Debug mode
    if cfg.debug:
        trainer_args["fast_dev_run"] = True

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Setup callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5)

    logger = CometLogger(
        api_key=find_comet_api_key(),
        project_name=f"BioMedKG-GCL-{cfg.data.node_type}",
        save_dir=log_dir,
        experiment_name=log_name,
    )

    trainer_args.update(
        {
            "max_epochs": cfg.epochs,
            "check_val_every_n_epoch": cfg.val_every_epoch,
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
    )

    test_args = {
        "model": model,
        "dataloaders": data_module.test_dataloader(),
    }

    if not cfg.debug:
        test_args["ckpt_path"] = "best"

    trainer.test(**test_args)


if __name__ == "__main__":
    main()

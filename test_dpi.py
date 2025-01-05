import hydra
from hydra.utils import instantiate
from lightning.pytorch import Trainer, seed_everything
from omegaconf import DictConfig

from biomedkg.kge_module import KGEModule


@hydra.main(version_base=None, config_path="configs", config_name="dpi")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    data_module = instantiate(
        cfg.data, gcl_model=cfg.gcl_model, gcl_fuse_method=cfg.gcl_fuse_method
    )
    data_module.setup(stage="split")

    print("=" * 20)
    print(f"Load from checkpoint: {cfg.pretrained_path}")
    print("=" * 20)

    model = KGEModule.load_from_checkpoint(cfg.pretrained_path)

    model.neg_ratio = cfg.neg_ratio

    model.edge_mapping = data_module.edge_map_index

    trainer_args = {
        "accelerator": "auto",
        "log_every_n_steps": 10,
        "deterministic": True,
        "devices": cfg.devices,
    }

    print("=" * 20)
    print(f"Neg Ratio: {model.neg_ratio}")
    print("=" * 20)

    trainer = Trainer(**trainer_args)

    test_args = {
        "model": model,
        "dataloaders": data_module.test_dataloader(loader_type="saint"),
    }

    trainer.test(**test_args)


if __name__ == "__main__":
    main()

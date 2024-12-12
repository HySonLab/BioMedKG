import torch_geometric.transforms as T
from lightning import LightningDataModule
from torch_geometric.loader import GraphSAINTRandomWalkSampler, NeighborLoader

from biomedkg.data import dataset, node


def get_node_encode_method(
    node_init_method: str,
    embed_dim: int,
    model_name: str,
    fuse_method: str,
    modality_config_path: str,
):
    if node_init_method is None or node_init_method == "random":
        return node.RandomEncode(embed_dim=embed_dim)
    elif node_init_method == "lm":
        return node.LMMultiModalsEncode(
            config_file=modality_config_path, embed_dim=embed_dim
        )
    elif node_init_method == "gcl":
        return node.GCLEncode(
            model_name=model_name, fuse_method=fuse_method, embed_dim=embed_dim
        )


class PrimeKGModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        embed_dim: int,
        node_type: list[str],
        batch_size: int,
        val_ratio: float,
        test_ratio: float,
        node_init_method: str = None,
        gcl_model: str = None,
        gcl_fuse_method: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.node_type = node_type
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size

        self.encoder = get_node_encode_method(
            node_init_method=node_init_method,
            embed_dim=embed_dim,
            model_name=gcl_model,
            fuse_method=gcl_fuse_method,
            modality_config_path="configs/lm_modality/primekg_modality.yaml",
        )

    def setup(self, stage: str = "split"):
        self.primekg = dataset.PrimeKG(
            data_dir=self.data_dir, node_type=self.node_type, encoder=self.encoder
        )
        self.edge_map_index = self.primekg.edge_map_index

        self.data = self.primekg.data

        if stage == "split":
            self.train_data, self.val_data, self.test_data = T.RandomLinkSplit(
                num_val=self.val_ratio,
                num_test=self.test_ratio,
                neg_sampling_ratio=0.0,
            )(data=self.data)

    def subgraph_dataloader(
        self,
    ):
        return NeighborLoader(
            data=self.data,
            num_neighbors=[-1],
            num_workers=0,
            shuffle=False,
        )

    def all_dataloader(self):
        return NeighborLoader(
            data=self.data,
            batch_size=self.batch_size,
            num_neighbors=[30] * 3,
            num_workers=0,
        )

    def train_dataloader(self, loader_type: str = "neighbor"):
        assert loader_type in ["neighbor", "saint"]

        if loader_type == "neighbor":
            return NeighborLoader(
                data=self.train_data,
                batch_size=self.batch_size,
                num_neighbors=[30] * 3,
                num_workers=0,
                shuffle=True,
            )
        elif loader_type == "saint":
            return GraphSAINTRandomWalkSampler(
                data=self.train_data,
                batch_size=self.batch_size,
                walk_length=10,
                num_steps=1000,
                num_workers=0,
            )

    def val_dataloader(self, loader_type: str = "neighbor"):
        assert loader_type in ["neighbor", "saint"]

        if loader_type == "neighbor":
            return NeighborLoader(
                data=self.val_data,
                batch_size=self.batch_size,
                num_neighbors=[30] * 3,
                num_workers=0,
            )
        elif loader_type == "saint":
            return GraphSAINTRandomWalkSampler(
                data=self.val_data,
                batch_size=self.batch_size,
                walk_length=10,
                num_steps=100,
                num_workers=0,
            )

    def test_dataloader(self, loader_type: str = "neighbor"):
        assert loader_type in ["neighbor", "saint"]

        if loader_type == "neighbor":
            return NeighborLoader(
                data=self.test_data,
                batch_size=self.batch_size,
                num_neighbors=[30] * 3,
                num_workers=0,
            )
        elif loader_type == "saint":
            return GraphSAINTRandomWalkSampler(
                data=self.test_data,
                batch_size=self.batch_size,
                walk_length=10,
                num_steps=100,
                num_workers=0,
            )


class DPIModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        embed_dim: int,
        batch_size: int,
        val_ratio: float,
        test_ratio: float,
        node_init_method: str = None,
        gcl_model: str = None,
        gcl_fuse_method: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size

        self.encoder = get_node_encode_method(
            node_init_method=node_init_method,
            embed_dim=embed_dim,
            model_name=gcl_model,
            fuse_method=gcl_fuse_method,
            modality_config_path="configs/lm_modality/dpi_modality.yaml",
        )

    def setup(self, stage: str = "split"):
        self.dpi = dataset.DPI(data_dir=self.data_dir, encoder=self.encoder)
        self.edge_map_index = self.dpi.edge_map_index
        self.data = T.ToUndirected()(self.dpi.data)

        if stage == "split":
            self.train_data, self.val_data, self.test_data = T.RandomLinkSplit(
                num_val=self.val_ratio,
                num_test=self.test_ratio,
                neg_sampling_ratio=0.0,
            )(data=self.data)

    def subgraph_dataloader(self):
        return NeighborLoader(
            data=self.data,
            num_neighbors=[-1],
            num_workers=0,
            shuffle=False,
        )

    def all_dataloader(self):
        return NeighborLoader(
            data=self.data,
            batch_size=self.batch_size,
            num_neighbors=[30] * 3,
            num_workers=0,
        )

    def train_dataloader(self, loader_type: str = "neighbor"):
        assert loader_type in ["neighbor", "saint"]

        if loader_type == "neighbor":
            return NeighborLoader(
                data=self.train_data,
                batch_size=self.batch_size,
                num_neighbors=[30] * 3,
                num_workers=0,
                shuffle=True,
            )
        elif loader_type == "saint":
            return GraphSAINTRandomWalkSampler(
                data=self.train_data,
                batch_size=self.batch_size,
                walk_length=10,
                num_steps=1000,
                num_workers=0,
            )

    def val_dataloader(self, loader_type: str = "neighbor"):
        assert loader_type in ["neighbor", "saint"]

        if loader_type == "neighbor":
            return NeighborLoader(
                data=self.val_data,
                batch_size=self.batch_size,
                num_neighbors=[30] * 3,
                num_workers=0,
            )
        elif loader_type == "saint":
            return GraphSAINTRandomWalkSampler(
                data=self.val_data,
                batch_size=self.batch_size,
                walk_length=10,
                num_steps=100,
                num_workers=0,
            )

    def test_dataloader(self, loader_type: str = "neighbor"):
        assert loader_type in ["neighbor", "saint"]

        if loader_type == "neighbor":
            return NeighborLoader(
                data=self.test_data,
                batch_size=self.batch_size,
                num_neighbors=[30] * 3,
                num_workers=0,
            )
        elif loader_type == "saint":
            return GraphSAINTRandomWalkSampler(
                data=self.test_data,
                batch_size=self.batch_size,
                walk_length=10,
                num_steps=100,
                num_workers=0,
            )

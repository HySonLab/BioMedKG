from lightning import LightningDataModule
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from typing import Callable

from biomedkg.modules.data import PrimeKG

class PrimeKGModule(LightningDataModule):
    def __init__(
            self, 
            data_dir : str, 
            process_node_lst : set[str],
            process_edge_lst : set[str] = {},
            batch_size : int = 64,
            val_ratio : float = 0.05,
            test_ratio : float = 0.15,
            encoder : Callable = None
            ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.process_node_lst = process_node_lst
        self.process_edge_lst = process_edge_lst
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.encoder = encoder

    def setup(self, stage : str = "split"):
        self.primekg = PrimeKG(
            data_dir=self.data_dir,
            process_node_lst=self.process_node_lst,
            process_edge_lst=self.process_edge_lst,
            encoder=self.encoder
        )

        self.data = self.primekg.get_data()

        if stage == "split":
            self.train_data, self.val_data, self.test_data = T.RandomLinkSplit(
                num_val=self.val_ratio,
                num_test=self.test_ratio,
                neg_sampling_ratio=0.,
            )(data=self.data)
    
    def subgraph_dataloader(self,):
        return NeighborLoader(
            data=self.data,
            batch_size=self.batch_size,
            num_neighbors=[-1],
            num_workers=0,
        )
       
    def all_dataloader(self):
        return NeighborLoader(
            data=self.data,
            batch_size=self.batch_size,
            num_neighbors=[30] * 3,
            num_workers=0,
        )

    def train_dataloader(self):
        return NeighborLoader(
            data=self.train_data,
            batch_size=self.batch_size,
            num_neighbors=[30] * 3,
            num_workers=0,
            shuffle=True,
        )

    def val_dataloader(self):
        return NeighborLoader(
            data=self.val_data,
            batch_size=self.batch_size,
            num_neighbors=[30] * 3,
            num_workers=0,
        )

    def test_dataloader(self):
        return NeighborLoader(
            data=self.test_data,
            batch_size=self.batch_size,
            num_neighbors=[30] * 3,
            num_workers=0,
        )
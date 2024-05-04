from lightning import LightningDataModule
import torch_geometric.transforms as T
from torch_geometric.loader import GraphSAINTRandomWalkSampler, NeighborLoader
from typing import Callable

from biomedkg.modules.data import PrimeKG

class PrimeKGModule(LightningDataModule):
    def __init__(
            self, 
            data_dir : str, 
            process_node_lst : set[str],
            process_edge_lst : set[str],
            batch_size : int,
            val_ratio : float,
            test_ratio : float,
            num_steps : int,
            encoder : Callable = None
            ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.process_node_lst = process_node_lst
        self.process_edge_lst = process_edge_lst
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.encoder = encoder

    def setup(self, stage : str = None):
        primekg = PrimeKG(
            data_dir=self.data_dir,
            process_node_lst=self.process_node_lst,
            process_edge_lst=self.process_edge_lst,
            encoder=self.encoder
        )

        self.data = primekg.get_data()

        self.train_data, self.val_data, self.test_data = T.RandomLinkSplit(
            num_val=self.val_ratio,
            num_test=self.test_ratio,
            neg_sampling_ratio=0.,
        )(data=self.data)
       

    def train_dataloader(self):
        return GraphSAINTRandomWalkSampler(
            data = self.train_data,
            batch_size=self.batch_size,
            walk_length=20,
            num_steps=self.num_steps,
            sample_coverage=100,
            num_workers=0
        )

    def val_dataloader(self):
        return GraphSAINTRandomWalkSampler(
            data = self.val_data,
            batch_size=self.batch_size,
            walk_length=20,
            num_steps=self.num_steps,
            sample_coverage=100,
            num_workers=0
        )

    def test_dataloader(self):
        return GraphSAINTRandomWalkSampler(
            data = self.test_data,
            batch_size=self.batch_size,
            walk_length=20,
            num_steps=self.num_steps,
            sample_coverage=100,
            num_workers=0
        )
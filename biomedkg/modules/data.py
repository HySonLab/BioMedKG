import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch_geometric.data import HeteroData

from biomedkg.modules.utils import clean_name
from biomedkg.configs.gcl import gcl_settings

class PrimeKG:

    def __init__(
            self, 
            data_dir : str,
            process_node_lst : set[str],
            process_edge_lst : set[str],
            encoder : dict = None):
        
        try:
            from tdc.resource import PrimeKG

            primekg = PrimeKG(path = data_dir)
            self.df = primekg.df
            
        except ModuleNotFoundError:
            csv_file = f"{data_dir}/kg.csv"
            
            if not os.path.exists(csv_file):
                os.system(f"wget -O {csv_file} https://dataverse.harvard.edu/api/access/datafile/6180620")

            self.df = pd.read_csv(csv_file, low_memory=False)

        if process_node_lst:
            self.df = self.df[self.df['x_type'].isin(list(process_node_lst)) & self.df['y_type'].isin(list(process_node_lst))]
        
        if process_edge_lst:
            self.df = self.df[self.df['relation'].isin(list(process_edge_lst))]

        self.data = HeteroData()

        self.list_nodes = self.df['x_type'].unique()
        self.list_edges = self.df['relation'].unique()

        print("\nList of node types: ")
        for node_name in self.list_nodes:
            print(f"\t- {node_name}")

        print(f"\nList of edge types:")
        for edge_type in self.list_edges:
            print(f"\t- {edge_type}")

        self.encoder = encoder

    def get_data(self,):
        self._build_node_embedding()
        self._build_edge_index()
        return self.data.to_homogeneous()
    

    def _build_node_embedding(self,):
        self.mapping_dict = dict()

        for node_type in tqdm(self.list_nodes, desc="Load node"):
            node_type_df = self.df[self.df['x_type'] == node_type]['x_name'].unique()

            lst_node_name = sorted(list(node_type_df))
            
            node_mapping = {node_name: index for index, node_name in enumerate(lst_node_name)}

            self.mapping_dict[node_type] = node_mapping
            
            if self.encoder is not None:
                embedding = self.encoder(lst_node_name)
            else:
                # Only random init on GCL traning
                print("\033[94m" + "Random initialize node embedding..." + "\033[0m")

                embedding = torch.empty(len(lst_node_name), gcl_settings.GCL_IN_DIMS)
                embedding = torch.nn.init.xavier_normal(embedding)

            node_type = clean_name(node_type)
            self.data[node_type].x = embedding
        
    def _build_edge_index(self,):
        for relation_type in tqdm(self.list_edges, desc="Load edge"):
            relation_df = self.df[self.df['relation'] == relation_type][['x_type', 'x_name', 'relation', 'y_type', 'y_name']]
            triples = relation_df[["x_type", "relation", "y_type"]].drop_duplicates().values

            head, relation, tail = triples[0]

            node_pair_df = relation_df[
                (self.df['x_type'] == head) & (self.df['y_type'] == tail)
                ][['x_name', 'y_name']]
            
            src = [self.mapping_dict[head][index] for index in node_pair_df['x_name']]
            dst = [self.mapping_dict[tail][index] for index in node_pair_df['y_name']]
            
            edge_index = torch.tensor([src, dst])

            head = clean_name(head)
            tail = clean_name(tail)
            relation = clean_name(relation)

            self.data[head, relation, tail].edge_index = edge_index
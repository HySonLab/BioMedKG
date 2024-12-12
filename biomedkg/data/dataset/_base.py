from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm.auto import tqdm

from biomedkg.common import clean_name


class TripletBase:
    def __init__(
        self,
        df: pd.DataFrame,
        encoder: Callable,
    ):
        self.df = df
        self.encoder = encoder

        self.data, self.edge_map_index, self.node_list = self.construct_hetero_data()

    def construct_hetero_data(self) -> tuple:
        all_node_name = list()

        list_nodes = np.unique(
            np.concatenate([self.df["x_type"].unique(), self.df["y_type"].unique()])
        )
        list_edges = self.df["relation"].unique()

        data = HeteroData()

        node_to_index, index_to_edge = dict(), dict()

        for node_type in tqdm(list_nodes, desc="Load node"):
            node_df_x = self.df[self.df["x_type"] == node_type]
            node_df_y = self.df[self.df["y_type"] == node_type]
            lst_node_name = set(node_df_x["x_name"].values) | set(
                node_df_y["y_name"].values
            )

            # Construct the node to index mapping
            _mapping = dict()
            lst_node_name: list[str] = sorted(lst_node_name)
            all_node_name.extend(lst_node_name)

            for index, node_name in enumerate(lst_node_name):
                _mapping[node_name] = index

            _mapping = {
                node_name: index for index, node_name in enumerate(lst_node_name)
            }
            node_to_index[node_type] = _mapping

            # Get the embedding for that node
            embedding = self.encoder(lst_node_name)

            print(f"Random Init node ratio is {self.encoder.random_init_ratio}")

            node_type = clean_name(node_type)
            data[node_type].x = embedding

        for edge_id, relation_type in enumerate(tqdm(list_edges, desc="Load edge")):
            relation_df = self.df[self.df["relation"] == relation_type][
                ["x_type", "x_name", "relation", "y_type", "y_name"]
            ]
            triples = (
                relation_df[["x_type", "relation", "y_type"]].drop_duplicates().values
            )

            head, relation, tail = triples[0]

            node_pair_df = relation_df[
                (self.df["x_type"] == head) & (self.df["y_type"] == tail)
            ][["x_name", "y_name"]]

            src = [node_to_index[head][name] for name in node_pair_df["x_name"]]
            dst = [node_to_index[tail][name] for name in node_pair_df["y_name"]]

            edge_index = torch.Tensor([src, dst])

            head = clean_name(head)
            tail = clean_name(tail)
            relation = clean_name(relation)

            data[head, relation, tail].edge_index = edge_index.to(torch.long)
            index_to_edge[edge_id] = relation_type

        return data.to_homogeneous(), index_to_edge, all_node_name

import os
import torch
import pandas as pd
from tqdm.auto import tqdm
from torch_geometric.data import HeteroData

from biomedkg.modules.utils import clean_name

class TripletBase:
    def __init__(
            self, 
            df : pd.DataFrame,
            embed_dim : int,
            encoder : dict = None,
            ):
        self.df = df
        self.list_nodes = self.df['x_type'].unique()
        self.list_edges = self.df['relation'].unique()
        self.encoder = encoder
        self.embedding_dim = embed_dim

        self.data = HeteroData()

    def get_data(self,):
        self._build_node_embedding()
        self._build_edge_index()
        return self.data.to_homogeneous()
    

    def _build_node_embedding(self,):
        self.modality_mapping = dict()
        self.mapping_dict = dict()

        node_id = 0

        for node_type in tqdm(self.list_nodes, desc="Load node"):
            node_df = self.df[self.df['x_type'] == node_type]
            lst_node_name = set(node_df['x_name'].values)

            node_mapping = dict()
            lst_node_name = sorted(lst_node_name)
            for index, node_name in enumerate(lst_node_name):
                node_mapping[node_name] = index
                self.mapping_dict[node_name] = node_id
                node_id += 1
                
            node_mapping = {node_name: index for index, node_name in enumerate(lst_node_name)}

            self.modality_mapping[node_type] = node_mapping
            
            if self.encoder is not None:
                embedding = self.encoder(lst_node_name)
            else:
                assert self.embedding_dim is not None

                embedding = torch.empty(len(lst_node_name), self.embedding_dim)
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
            
            src = [self.modality_mapping[head][index] for index in node_pair_df['x_name']]
            dst = [self.modality_mapping[tail][index] for index in node_pair_df['y_name']]
            
            edge_index = torch.tensor([src, dst])

            head = clean_name(head)
            tail = clean_name(tail)
            relation = clean_name(relation)

            self.data[head, relation, tail].edge_index = edge_index


class PrimeKG(TripletBase):
    def __init__(
            self, 
            data_dir : str,
            process_node_lst : set[str],
            process_edge_lst : set[str],
            embed_dim : int,
            encoder : dict = None,
            ):
        
        try:
            from tdc.resource import PrimeKG

            primekg = PrimeKG(path = data_dir)
            df = primekg.df
            
        except ModuleNotFoundError:
            csv_file = f"{data_dir}/kg.csv"
            
            if not os.path.exists(csv_file):
                os.system(f"wget -O {csv_file} https://dataverse.harvard.edu/api/access/datafile/6180620")

            df = pd.read_csv(csv_file, low_memory=False)

        if process_node_lst:
            df = df[df['x_type'].isin(list(process_node_lst)) & df['y_type'].isin(list(process_node_lst))]
        
        if process_edge_lst:
            df = df[df['relation'].isin(list(process_edge_lst))]

        super().__init__(df=df, embed_dim=embed_dim, encoder=encoder)


class BioKG(TripletBase):
    def __init__(
            self,
            data_dir : str,
            embed_dim : int,
            encoder : dict = None
            ):
        # Prepare pd.DataFrame with 5 columns ['x_type', 'x_name', 'relation', 'y_type', 'y_name']
        df = pd.read_csv(data_dir, delimiter="\t", names=["x_name", "relation", "y_name"])

        # Initialize lists to store x_type and y_type
        x_types = []
        y_types = []

        # Iterate through the DataFrame to determine x_type and y_type
        for i in range(len(df)):
            relation = df.iloc[i].relation

            # Determine the types based on the relation
            if "_" in relation:
                x_type, y_type = map(str.lower, relation.split("_")[:2])
            else:
                x_type = "gene/protein" if relation[0] == "P" else "drug"
                y_type = "gene/protein" if relation[1] == "P" else "drug"

            # Append the types to the lists
            x_types.append(x_type)
            y_types.append(y_type)

        # Add the new columns to the DataFrame
        df['x_type'] = x_types
        df['y_type'] = y_types

        super().__init__(df=df, embed_dim=embed_dim, encoder=encoder)

if __name__ == "__main__":
    biokg = BioKG("data/biokg.links-test.csv", 768)

    print(biokg.df.head())

    print("Test successful!")
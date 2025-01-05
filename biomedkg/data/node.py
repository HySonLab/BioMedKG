import glob
import os
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from biomedkg import data_module, gcl_module
from biomedkg.data.embed import NodeEmbedding
from biomedkg.kge_module import KGEModule


class LMMultiModalsEncode:
    def __init__(self, config_file: str, embed_dim: int = 768, batch_size: int = 128):
        self.conf = OmegaConf.load(config_file)
        self.artifact_path = os.path.join(
            "data", "embed", f"{Path(config_file).stem}_lm.pickle"
        )
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.node_mapping = self.load()
        self.random_init_ratio = 0

    def __call__(self, lst_node: List[str]) -> torch.Tensor:
        random_init = 0
        node_embedding = []

        for node_name in lst_node:
            embedding = self.node_mapping.get(node_name, None)
            if embedding is None:
                embedding = torch.nn.init.xavier_normal_(torch.empty(2, self.embed_dim))
                random_init += 1
            node_embedding.append(torch.tensor(embedding))

        node_embedding = torch.stack(node_embedding, dim=0)
        self.random_init_ratio = random_init / len(lst_node)
        return node_embedding.to(torch.float)

    def load(self) -> dict[str, np.array]:
        if not os.path.exists(self.artifact_path):
            self._get_embeddings()
        with open(self.artifact_path, "rb") as file:
            node_embedding = pickle.load(file)
        return node_embedding

    def _get_embeddings(self):
        node_mapping = dict()
        for node_type in self.conf.keys():
            if self.conf[node_type].get("file_name", None) is None:
                for sub_node_type in self.conf[node_type]:
                    feature_dict = self._get_feature_dict(
                        **self.conf[node_type][sub_node_type]
                    )
                    node_mapping.update(feature_dict)
            else:
                feature_dict = self._get_feature_dict(**self.conf[node_type])
                node_mapping.update(feature_dict)

        with open(self.artifact_path, "wb") as file:
            pickle.dump(node_mapping, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _get_feature_dict(
        self,
        file_name: str,
        idetifier_column: str,
        modality_columns: List[str],
        model_name_for_each_modality: List[str],
    ) -> dict[str, np.array]:

        df = pd.read_csv(file_name)
        df = df[[idetifier_column] + modality_columns]
        df = df.drop_duplicates(keep="first")

        model_dict = dict()
        for modality, model_name in zip(modality_columns, model_name_for_each_modality):
            model_dict[modality] = NodeEmbedding(model_name_or_path=model_name)

        feature_dict = dict()
        for idx in tqdm(range(0, len(df), self.batch_size)):
            row = df.iloc[idx : idx + self.batch_size]

            all_embeddings = list()
            for modality in modality_columns:
                modality_values = row[modality].to_list()

                is_nan_mask = pd.isna(modality_values)

                random_embeddings = torch.nn.init.xavier_normal_(
                    torch.empty(np.sum(is_nan_mask), self.embed_dim)
                )

                non_nan_values = [
                    modality_values[i]
                    for i in range(len(modality_values))
                    if not is_nan_mask[i]
                ]
                if len(non_nan_values) != 0:
                    valid_embeddings = model_dict[modality](non_nan_values)

                combined_embeddings = np.empty((len(row), self.embed_dim))
                combined_embeddings[is_nan_mask] = random_embeddings

                if len(non_nan_values) != 0:
                    combined_embeddings[~is_nan_mask] = valid_embeddings

                all_embeddings.append(combined_embeddings)

            embeddings = np.stack(all_embeddings, axis=1)

            # Normalize the embeddings across modalities
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / norms

            feature_dict.update(
                dict(
                    zip(
                        row[idetifier_column].to_list(),
                        [
                            normalized_embeddings[i]
                            for i in range(normalized_embeddings.shape[0])
                        ],
                    )
                )
            )

        del model_dict
        import gc

        gc.collect()

        return feature_dict


class RandomEncode:
    def __init__(
        self,
        embed_dim: int = 768,
    ):
        self.embed_dim = embed_dim
        self.random_init_ratio = 1

    def __call__(self, lst_node: List[str]) -> torch.Tensor:
        node_embedding = torch.nn.init.xavier_normal_(
            torch.empty(len(lst_node), self.embed_dim)
        )

        return node_embedding


class GCLEncode:
    data_gcl = os.path.join("data", "gcl_embed")
    os.makedirs(data_gcl, exist_ok=True)

    gcl_ckpt = os.path.join("ckpt", "gcl")
    assert os.path.exists(gcl_ckpt), "Can't find checkpoints from {gcl_ckpt}"

    def __init__(self, model_name: str, fuse_method: str, embed_dim: int):
        self.model_name = model_name
        self.fuse_method = fuse_method
        self.embed_dim = embed_dim
        self.artifact_path = os.path.join(
            self.data_gcl, f"{model_name}_{fuse_method}.pickle"
        )
        self.node_mapping = self.load()
        self.random_init_ratio = 0

    def __call__(self, lst_node: List[str]) -> torch.Tensor:
        node_embedding = []
        random_init = 0
        for node_name in lst_node:
            embedding = self.node_mapping.get(node_name, None)
            if embedding is None:
                embedding = torch.nn.init.xavier_normal_(torch.empty(1, self.embed_dim))
                random_init += 1
            node_embedding.append(torch.tensor(embedding))

        node_embedding = torch.stack(node_embedding, dim=0)
        self.random_init_ratio = random_init / len(lst_node)
        return node_embedding

    def load(self) -> dict[str, np.array]:
        if not os.path.exists(self.artifact_path):
            self._get_embeddings()
        with open(self.artifact_path, "rb") as file:
            node_embedding = pickle.load(file)
        return node_embedding

    def _get_embeddings(self):
        node_mapping = dict()

        for node_type in ["gene", "drug", "disease"]:
            pattern = f"{self.gcl_ckpt}/{node_type}/{self.model_name}*{self.fuse_method}*lm*/*.ckpt"
            all_files = glob.glob(pattern)

            assert len(all_files) != 0, f"Can't find checkpoint with pattern {pattern}"

            ckpt_path = all_files[0]

            if self.model_name == "dgi":
                model = gcl_module.DGIModule.load_from_checkpoint(ckpt_path)
            elif self.model_name == "grace":
                model = gcl_module.GRACEModule.load_from_checkpoint(ckpt_path)
            elif self.model_name == "ggd":
                model = gcl_module.GGDModule.load_from_checkpoint(ckpt_path)
            else:
                raise NotImplementedError

            if node_type.startswith("gene"):
                node_type = "gene/protein"

            data_args = {
                "data_dir": "./data/primekg",
                "embed_dim": 768,
                "node_type": [node_type],
                "batch_size": 128,
                "val_ratio": 0.2,
                "test_ratio": 0.2,
                "node_init_method": "lm",
            }

            data = data_module.PrimeKGModule(**data_args)
            data.setup(stage="split")

            node_list = data.primekg.node_list
            dataloader = data.subgraph_dataloader()

            for nodes, batch in tqdm(zip(node_list, dataloader), total=len(dataloader)):
                batch = batch.to(model.device)
                with torch.no_grad():
                    out = model(batch.x, batch.edge_index)
                    out = out.detach().cpu().numpy()[: batch.batch_size]

                node_mapping[nodes] = out

        with open(self.artifact_path, "wb") as file:
            pickle.dump(node_mapping, file, protocol=pickle.HIGHEST_PROTOCOL)


class KGEEncode:
    def __init__(
        self,
        ckpt_path: str,
        node_init_method: str,
        gcl_model: str,
        gcl_fuse_method: str,
        out_dim: int = 256,
    ):
        self.ckpt_path = ckpt_path
        self.node_init_method = node_init_method
        self.gcl_model = gcl_model
        self.gcl_fuse_method = gcl_fuse_method
        self.out_dim = out_dim

        save_dir = os.path.join("data", "kge_embed")
        os.makedirs(save_dir, exist_ok=True)

        save_file_name = "_".join(ckpt_path.split("/")[-2:]).split(".")[0]
        self.artifact_path = os.path.join(save_dir, save_file_name)

        self.node_mapping = self.load()

    def __call__(self, lst_node: List[str]) -> torch.Tensor:
        node_embedding = []
        for node_name in lst_node:
            embedding = self.node_mapping.get(node_name, None)
            if embedding is None:
                embedding = torch.nn.init.xavier_normal_(torch.empty(1, self.out_dim))
            node_embedding.append(torch.tensor(embedding))

        node_embedding = torch.stack(node_embedding, dim=0)

        return node_embedding

    def load(
        self,
    ):
        if not os.path.exists(self.artifact_path):
            self._get_embeddings()
        with open(self.artifact_path, "rb") as file:
            node_embedding = pickle.load(file)
        return node_embedding

    def _get_embeddings(self):
        node_mapping = dict()

        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError

        model = KGEModule.load_from_checkpoint(self.ckpt_path)

        if self.node_init_method in ["random", "lm"]:
            in_dim = 768
        else:
            in_dim = 256

        self.embed_dim = in_dim

        data_args = {
            "data_dir": "./data/primekg",
            "embed_dim": in_dim,
            "node_type": ["gene/protein", "drug", "disease"],
            "batch_size": 64,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "node_init_method": self.node_init_method,
            "gcl_model": self.gcl_model,
            "gcl_fuse_method": self.gcl_fuse_method,
        }

        data = data_module.PrimeKGModule(**data_args)
        data.setup()

        node_list = data.primekg.node_list
        dataloader = data.subgraph_dataloader()

        for nodes, batch in tqdm(zip(node_list, dataloader), total=len(dataloader)):
            batch = batch.to(model.device)
            with torch.no_grad():
                out = model(batch.x, batch.edge_index, batch.edge_type)
                out = out.detach().cpu().numpy()[: batch.batch_size]

            node_mapping[nodes] = out

        with open(self.artifact_path, "wb") as file:
            pickle.dump(node_mapping, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    encoder = LMMultiModalsEncode(
        config_file="../../../configs/lm_modality/primekg_modality.yaml"
    )

    node_name_lst = [
        "(1,2,6,7-3H)Testosterone",
        "(4-{(2S)-2-[(tert-butoxycarbonyl)amino]-3-methoxy-3-oxopropyl}phenyl)methaneseleninic acid",
        "(6R)-Folinic acid",
        "(6S)-5,6,7,8-tetrahydrofolic acid",
        "(R)-warfarin",
        "(S)-2-Amino-3-(4h-Selenolo[3,2-B]-Pyrrol-6-Yl)-Propionic Acid",
        "(S)-2-Amino-3-(6h-Selenolo[2,3-B]-Pyrrol-4-Yl)-Propionic Acid",
        "(S)-Warfarin",
        "1,10-Phenanthroline",
        "1-Testosterone",
    ]

    embeddings = encoder(node_name_lst)

    print(embeddings.size())

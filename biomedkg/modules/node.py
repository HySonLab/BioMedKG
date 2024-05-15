import os
import glob
import torch
import pickle
from typing import List
from pathlib import Path
from biomedkg.configs import kge_settings

class EncodeNode:
    def __init__(
        self,
        embed_path : str,
        ):
        assert os.path.exists(embed_path), f"Can't find {embed_path}."

        self.embed_path = embed_path

        self.node_embed = self._get_feature_embedding_dict()
    
    def __call__(self, node_name_lst:List[str]) -> torch.tensor:
        node_embedding = []

        for node_name in node_name_lst:

            # Get embeddings by name
            modality_embedding = self.node_embed.get(node_name, None)

            if modality_embedding is not None:
                node_embedding.append(torch.tensor(modality_embedding))
            else:
                print("\033[93m" + f"{node_name}" + "\033[0m" +  " not found.")
                node_embedding.append(torch.rand(kge_settings.KGE_IN_DIMS))
                    
        node_embedding = torch.stack(node_embedding, dim=0)
        return node_embedding

    def _get_feature_embedding_dict(self,):
        node_embed = dict()

        pickle_files = glob.glob(self.embed_path + "/*.pickle")
        for pickle_file in pickle_files:
            with open(pickle_file, "rb") as file:
                data = pickle.load(file=file)
                node_embed.update(data)
        return node_embed


class EncodeNodeWithModality:
    def __init__(
        self,
        entity_type : str,
        embed_path : str,
        ):
        assert os.path.exists(embed_path), f"Can't find {embed_path}."
        assert entity_type in ["gene", "disease", "drug"]

        self.entity_type = entity_type
        self.embed_path = embed_path

        self.modality_dict = self._get_feature_embedding_dict()
    
    def __call__(self, node_name_lst:List[str]) -> torch.tensor:
        node_embedding = []

        embedding_max_length = 0

        for node_name in node_name_lst:
            fused_embedding = []

            # Ensure that "sequence" is always in the second position
            modality_sorted_lst = sorted(self.modality_dict.keys(), key=lambda s: (1, s) if "seq" in s else (0, s))

            # Get embeddings by name
            for modality in modality_sorted_lst:

                modality_embedding = self.modality_dict[modality].get(node_name, None)

                if modality_embedding is not None:
                    # Find the embedding size if there are missing modalities
                    if len(modality_embedding) > embedding_max_length:
                        embedding_max_length = len(modality_embedding)
                    fused_embedding.append(modality_embedding)
                else:
                    print("\033[93m" + f"{node_name}" + "\033[0m" +  " not found.")
                    fused_embedding.append([])
                    
            # Random initialization if missing modality, then convert it to a tensor
            for idx in range(len(fused_embedding)):
                if len(fused_embedding[idx]) == 0:
                    fused_embedding[idx] = torch.rand(embedding_max_length)
                else:
                    fused_embedding[idx] = torch.tensor(fused_embedding[idx])
            
            node_embedding.append(torch.cat(fused_embedding, dim=0))

        node_embedding = torch.stack(node_embedding, dim=0)
        return node_embedding

    def _get_feature_embedding_dict(self,):
        modality_dict = dict()

        pickle_files = glob.glob(self.embed_path + f"/{self.entity_type}*.pickle")
        for pickle_file in pickle_files:
            modality_name = Path(pickle_file).stem.split("_")[1]
            with open(pickle_file, "rb") as file:
                data = pickle.load(file=file)
                modality_dict[modality_name] = data
        return modality_dict


if __name__ == "__main__":
    encoder = EncodeNodeWithModality(
        entity_type="gene",
        embed_path="../../data/embed"
    )

    node_name_lst = ['PHYHIP', 'GPANK1', 'ZRSR2','NRF1','PI4KA','SLC15A1','EIF3I','FAXDC2','MT1A','SORT1']

    embeddings = encoder(node_name_lst)

    print(embeddings.size())

    encoder = EncodeNode(
        embed_path="../../data/gcl_embed"
    )

    node_name_lst = ["(1,2,6,7-3H)Testosterone",
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

import os
import yaml
import pickle
import pandas as pd
from tqdm.auto import tqdm
from typing import List

from biomedkg.modules.utils import generator
from biomedkg.modules.node import NodeEmbedding

def get_feature(
        file_name : str,
        idetifier_column: str,
        modality_columns : List[str],
        model_name_for_each_modality : List[str],
        save_files : List[str],
        batch_size: int = 16,
):

    df = pd.read_csv(file_name)

    for modality, model_name, save_file in zip(modality_columns, model_name_for_each_modality, save_files):

        print(f"Process {modality} with {model_name}")
        
        sub_df = df[[idetifier_column, modality]]
        sub_df = sub_df.dropna()
        sub_df = sub_df.drop_duplicates()

        modality_ids = sub_df[idetifier_column].to_list()

        entity_name_lst = list()
        entity_feature_lst = list()

        node_embeder = NodeEmbedding(model_name_or_path=model_name)

        with tqdm(total=len(modality_ids), desc=f"Processing {modality}") as pbar:
            for identity in generator(modality_ids, batch_size):
                modality_feature = sub_df[sub_df[idetifier_column].isin(identity)]
                
                entity_names = modality_feature[idetifier_column].to_list()
                entity_features = modality_feature[modality].to_list()

                hidden_state = node_embeder(entity_features)
                
                entity_name_lst.extend(entity_names)
                entity_feature_lst.extend(hidden_state.numpy())
                
                pbar.update(len(identity))  
                        
        feature_dict = dict(zip(entity_name_lst, entity_feature_lst))
        
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))

        with open(save_file, "wb") as file:
            pickle.dump(feature_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

        del node_embeder

        import gc
        gc.collect()


if __name__ == "__main__":
    with open("modality.yaml", "r") as file:
        data = yaml.load(file, Loader=yaml.SafeLoader)
    
    for entity_type in data.keys():
        if entity_type == "gene/protein":
            gene_protein = data[entity_type]
            for gene_type in gene_protein.keys():
                entity = gene_protein[gene_type]
                get_feature(**entity)
        else:
            entity = data[entity_type]
            get_feature(**entity)
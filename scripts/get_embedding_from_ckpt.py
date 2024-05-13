import os
import json
import torch
import pickle
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from biomedkg import gcl_module
from biomedkg.data_module import PrimeKGModule
from biomedkg.modules.node import EncodeNodeWithModality
from biomedkg.configs import data_settings, train_settings
from lightning import seed_everything

def parse_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
            '--model_name', 
            type=str, 
            action='store', 
            choices=['dgi', 'grace', 'ggd'], 
            default='dgi', 
            help="Select contrastive model name")    
    
    parser.add_argument(
            '--ckpt', 
            type=str, 
            default=None,
            required=False,
            help="Path to ckpt file")
    opt = parser.parse_args()
    return opt


def main(
        model_name:str,
        ckpt:str,
):
    assert os.path.exists(ckpt)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if model_name == "dgi":
        model = gcl_module.DGIModule.load_from_checkpoint(ckpt)
    elif model_name == "grace":
        model = gcl_module.GRACEModule.load_from_checkpoint(ckpt)
    elif model_name == "ggd":
        model = gcl_module.GGDModule.load_from_checkpoint(ckpt)
    else:
        raise NotImplementedError

    seed_everything(train_settings.SEED)

    model = model.to(device)
    model.eval()
    
    json_file = os.path.join(os.path.dirname(ckpt), "node_mapping.json")

    with open(json_file, "r") as file:
        mapping_dict = json.load(file)

    for node_type, node_to_idx_dict in mapping_dict.items():
        assert isinstance(node_to_idx_dict, dict)
        assert isinstance(node_type, str)

        idx_to_node_dict = {v: k for k, v in node_to_idx_dict.items()}

        data_module = PrimeKGModule(
            data_dir=data_settings.DATA_DIR,
            process_node_lst=[node_type],
            process_edge_lst=data_settings.EDGES_LST,
            batch_size=train_settings.BATCH_SIZE,
            encoder=EncodeNodeWithModality(
                entity_type=node_type, 
                embed_path=os.path.join(os.path.dirname(data_settings.DATA_DIR),"embed")
                )
        )

        data_module.setup(stage="all")

        subgraph_loader = data_module.subgraph_dataloader()

        node_embedding_mapping = dict()

        for batch in tqdm(subgraph_loader):
            x = batch.x.to(device)

            with torch.no_grad():
                out = model(x, batch.edge_index.to(device))

            for node_id, embed in zip(batch.n_id[:batch.batch_size].tolist(), out[:batch.batch_size].detach().cpu().numpy()):
                node_embedding_mapping[idx_to_node_dict[node_id]] = embed

        save_dir = os.path.join(os.path.dirname(data_settings.DATA_DIR), "gcl_embed")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_file_name = os.path.join(
            save_dir,
            os.path.basename(os.path.dirname(ckpt)) + ".pickle",
        )

        with open(save_file_name, "wb") as file:
            pickle.dump(node_embedding_mapping, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Save {save_file_name} completed")


if __name__ == "__main__":
    main(**vars(parse_opt()))
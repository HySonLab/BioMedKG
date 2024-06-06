import os
import re
import json
import torch
import pickle
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from biomedkg import gcl_module, kge_module
from biomedkg.data_module import PrimeKGModule, BioKGModule
from biomedkg.modules.node import EncodeNodeWithModality, EncodeNode
from biomedkg.configs import data_settings, train_settings, node_settings, kge_settings
from biomedkg.modules.utils import find_device
from lightning import seed_everything


def parse_opt():
    parser = argparse.ArgumentParser()
   
    parser.add_argument(
            '--ckpt',
            type=str,
            default=None,
            required=False,
            help="Path to ckpt file")

    parser.add_argument(
        'data',
        type=str,
        default="primekg",
        required=True,
        choices=["primekg", "biokg"],
        help="Choose either primekg or biokg(both biokg and dpi_fda)"
    )
    opt = parser.parse_args()
    return opt


def main(
        ckpt:str,
        data:str
):
    assert os.path.exists(ckpt)


    device = find_device()


    seed_everything(train_settings.SEED)
   
    dir_name = os.path.dirname(ckpt)
    json_file = os.path.join(dir_name, "node_mapping.json")
    model_name = os.path.basename(dir_name).split("_")[0]
    modality_transform = os.path.basename(dir_name).split("_")[2]


    with open(json_file, "r") as file:
        mapping_dict = json.load(file)


    if re.search(r"gcl", dir_name):
        if model_name == "dgi":
            model = gcl_module.DGIModule.load_from_checkpoint(ckpt)
        elif model_name == "grace":
            model = gcl_module.GRACEModule.load_from_checkpoint(ckpt)
        elif model_name == "ggd":
            model = gcl_module.GGDModule.load_from_checkpoint(ckpt)
        else:
            raise NotImplementedError


        node_type = dir_name.split("_")[1]


        if node_type == "gene":
            process_node = ['gene/protein']
        else:
            process_node = [node_type]

        if data == "primekg":
            data_module = PrimeKGModule(
                data_dir=data_settings.DATA_DIR,
                process_node_lst=process_node,
                process_edge_lst=data_settings.EDGES_LST,
                batch_size=train_settings.BATCH_SIZE,
                encoder=EncodeNodeWithModality(
                    entity_type=node_type,
                    embed_path=os.path.join(os.path.dirname(data_settings.DATA_DIR),"embed")
                )
            )
        elif data == "biokg":
            data_module = BioKGModule(
                data_dir=data_settings.DATA_DIR,
                batch_size=train_settings.BATCH_SIZE,
                encoder=EncodeNodeWithModality(
                    entity_type=node_type,
                    embed_path=os.path.join(os.path.dirname(data_settings.DATA_DIR),"embed")
                )
            )
        else:
            raise NotImplementedError

        data_module.setup(stage="split", embed_dim=node_settings.PRETRAINED_NODE_DIM)
    else:
        model = kge_module.KGEModule.load_from_checkpoint(ckpt)


        # Decide node intialize method
        node_init_method = model.hparams.node_init_method


        if node_init_method == "gcl":
            encoder = EncodeNode(
                embed_path=os.path.join(os.path.dirname(data_settings.DATA_DIR), f"gcl_embed/{model_name}_{modality_transform}")
                )
        elif node_init_method == "llm":
            encoder = EncodeNodeWithModality(
                entity_type=list(data_settings.NODES_LST),
                embed_path=os.path.join(os.path.dirname(data_settings.DATA_DIR), "embed"),
                )
        elif node_init_method == "random":
            encoder = None
        else:
            raise NotImplementedError
       
        if node_init_method == "gcl" or node_init_method == "random":
            embed_dim = node_settings.GCL_TRAINED_NODE_DIM
        elif node_init_method == "llm":
            embed_dim = node_settings.PRETRAINED_NODE_DIM
        else:
            embed_dim = None


        # Setup data module
        if data == "primekg":
            data_module = PrimeKGModule(
                data_dir=data_settings.DATA_DIR,
                process_node_lst=data_settings.NODES_LST,
                process_edge_lst=data_settings.EDGES_LST,
                batch_size=train_settings.BATCH_SIZE,
                val_ratio=train_settings.VAL_RATIO,
                test_ratio=train_settings.TEST_RATIO,
                encoder=encoder,
            )
        elif data == "biokg":
            data_module = BioKGModule(
                data_dir=data_settings.DATA_DIR,
                batch_size=train_settings.BATCH_SIZE,
                val_ratio=train_settings.VAL_RATIO,
                test_ratio=train_settings.TEST_RATIO,
                encoder=encoder
            )
        else:
            raise NotImplementedError

        data_module.setup(stage="split", embed_dim=embed_dim)


    model = model.to(device)
    model.eval()


    subgraph_loader = data_module.subgraph_dataloader()


    idx_to_node_dict = {v: k for k, v in mapping_dict.items()}


    node_embedding_mapping = dict()


    for batch in tqdm(subgraph_loader):
        x = batch.x.to(device)


        with torch.no_grad():
            if re.search(r"gcl", dir_name):
                out = model(x, batch.edge_index.to(device))
            else:
                out = model(x, batch.edge_index.to(device), batch.edge_type.to(device))


        for node_id, embed in zip(batch.n_id[:batch.batch_size].tolist(), out[:batch.batch_size].detach().cpu().numpy()):
            node_embedding_mapping[idx_to_node_dict[node_id]] = embed


    if re.search(r"gcl", dir_name):
        save_dir = os.path.join(os.path.dirname(data_settings.DATA_DIR), f"gcl_embed/{model_name}_{modality_transform}")
    else:
        save_dir = os.path.join(os.path.dirname(data_settings.DATA_DIR), "kge_embed")


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

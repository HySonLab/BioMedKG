# import argparse

# import torch
# import torch.optim as optim
# import os 
# import torch
# from lightning.pytorch import seed_everything
# from biomedkg.configs import train_settings, data_settings, node_settings, kge_settings

# from tqdm import tqdm
# from biomedkg.modules.node import EncodeNode, EncodeNodeWithModality
# from biomedkg.data_module import BioKGModule

# from torch_geometric.nn import ComplEx, DistMult, TransE

# def parse_opt():
#         parser = argparse.ArgumentParser()

#         parser.add_argument(
#             '--model_name',
#             type=str, 
#             default="transe",   
#             choices=["transe", "dismult", "complex"],
#             help="model for training BioKG"
#         )

#         parser.add_argument(
#              '--gcl_embed_path',
#              type=str,
#              default=None,
#              required=False,
#              help="Path to your GCL embedding")

#         parser.add_argument(
#             '--k_in_hits@k',
#             type=int,
#             default=10,
#             required=True,
#             choices=[1, 3, 10],
#             help="k in hits@k"
#         )
        
#         opt = parser.parse_args()

#         return opt

# def main(model_name:str, kge_embed_path:str=None, k:int=10):
#     seed_everything(train_settings.SEED)

#     encoder = EncodeNode(
#         embed_path=kge_embed_path
#     )

#     embed_dim = node_settings.GCL_TRAINED_NODE_DIM

#     if model_name=="transe":
#         model=TransE
#     elif model_name=="complex":
#         model=ComplEx
#     elif model_name=="dismult":
#         model=DistMult

#     data_module = BioKGModule(
#         data_dir="data/biokg.links-test.csv",
#         id_name_dirs=["data/drug_idname.pkl", "data/disease_idname.pkl", "data/gene_idname.pkl"],
#         batch_size=train_settings.BATCH_SIZE,
#         val_ratio=train_settings.VAL_RATIO,
#         test_ratio=train_settings.TEST_RATIO,
#         encoder=encoder,
#     )
    
#     data_module.setup(stage="split", embed_dim=embed_dim)

#     model = model(
#         num_nodes=data_module.train_data.num_nodes,
#         num_relations=data_module.train_data.num_edge_types,
#         hidden_channels=kge_settings.KGE_HIDDEN_DIM,
#     )

#     loader = model.loader(
#         head_index=data_module.train_data.edge_index[0],
#         rel_type=data_module.train_data.edge_type,
#         tail_index=data_module.train_data.edge_index[1],
#         batch_size=train_settings.BATCH_SIZE,
#         shuffle=True,
#     )

#     optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)

#     def train():
#         model.train()
#         total_loss = total_examples = 0
#         for head_index, rel_type, tail_index in loader:
#             optimizer.zero_grad()
#             loss = model.loss(head_index, rel_type, tail_index)
#             loss.backward()
#             optimizer.step()
#             total_loss += float(loss) * head_index.numel()
#             total_examples += head_index.numel()
#         return total_loss / total_examples

#     @torch.no_grad()
#     def test(data):
#         model.eval()
#         return model.test(
#             head_index=data.edge_index[0],
#             rel_type=data.edge_type,
#             tail_index=data.edge_index[1],
#             batch_size=train_settings.BATCH_SIZE,
#             k=k, # Hits@k
#             log=True
#         )

#     for epoch in tqdm(range(1, train_settings.EPOCHS)):
#         loss = train()
#         print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
#         if epoch % 25 == 0:
#             rank, mrr, hits = test(data_module.val_data)
#             print(f'Epoch: {epoch:03d}'
#                 f'Val MRR: {mrr:.4f}, Val Hits@10: {hits:.4f}')

#     rank, mrr, hits_at_k = test(data_module.test_data)
#     print(f'Test Mean Rank: {rank:.2f}, Test MRR: {mrr:.4f}, '
#         f'Test Hits@{k}: {hits_at_k:.4f}')

# if __name__ == "__main__":
#     main(**vars(parse_opt()))

import argparse

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import EarlyStopping

from biomedkg.benchmark_module import BenchmarkModule
from biomedkg.modules.node import EncodeNode
from biomedkg.data_module import BioKGModule
from biomedkg.modules.utils import find_comet_api_key
from biomedkg.configs import train_settings

def parse_opt():
        parser = argparse.ArgumentParser()
        parser.add_argument(
                '--job', 
                type=str, 
                action='store', 
                choices=['finetune', 'zeroshot'], 
                default='finetune', 
                help="Do finetune or zeroshot benchmark"
            )
        
        parser.add_argument(
                '--path', 
                type=str, 
                action='store', 
                required=1,
                help="add kge checkpoint path"
        )
        
        opt = parser.parse_args()
        return opt
    
def main(job:str, path:str = None):
    seed_everything(train_settings.SEED)

    data_module = BioKGModule(
        data_dir="data/biokg-test.csv",
        batch_size=train_settings.BATCH_SIZE,
        test_ratio=train_settings.TEST_RATIO,
        encoder=None
    )

    data_module.setup(embed_dim=128)

    model = BenchmarkModule(
        model_path=path,
        out_dim=128,
        in_dim=128,
        hidden_dim=128,
        num_hidden_layers=2,        
        num_relation=8,
        scheduler_type=train_settings.SCHEDULER_TYPE,
        learning_rate=train_settings.LEARNING_RATE,
        warm_up_ratio=train_settings.WARM_UP_RATIO,
    )

    early_stopping = EarlyStopping(monitor="val_loss", mode="min")

    logger = CometLogger(
        api_key=find_comet_api_key(),
        project_name="BioMedKG-KGE",
        experiment_name="benchmark",
    )


    trainer_args = {
        "accelerator": "auto", 
        "log_every_n_steps": 10,
        "max_epochs": train_settings.EPOCHS,
        "enable_checkpointing": True, 
        "logger": logger, 
        "callbacks": [early_stopping], 
        "deterministic": True, 
        "gradient_clip_val": 1.0,
    }

    if isinstance(train_settings.DEVICES, list) and len(train_settings.DEVICES) > 1:
        trainer_args.update(
            {
                "devices": train_settings.DEVICES,
                "strategy": "ddp"
            }
        )
    else:
        if torch.cuda.device_count() > 0:
            trainer_args.update(
                {
                    "devices": train_settings.DEVICES,
                }
            )

    trainer = Trainer(**trainer_args)
    
    if job == "finetune":
        trainer.fit(
            model=model,
            train_dataloaders=data_module.train_dataloader(loader_type="graph_saint"),
        )
        trainer.test(
             model=model,
             dataloaders=data_module.testdataloader(loader_type="graph_saint"),
        )    
        
    elif job == "zeroshot":
        trainer.validate(
             model=model,
             dataloaders=data_module.val_dataloader(loader_type="graph_saint"),
        )
        trainer.test(
             model=model,
             dataloaders=data_module.test_dataloader(loader_type="graph_saint"),
        )


if __name__ == "__main__":
    main(**vars(parse_opt()))
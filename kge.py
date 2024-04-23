import torch_geometric.data
import torch_geometric.utils
from tqdm import tqdm

import torch
import torch_geometric
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GAE
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.utils import negative_sampling

from biomedkg.modules.encoder import RGCN
from biomedkg.modules.decoder import DistMult
from biomedkg.modules.utils import parameters_count

from biomedkg.configs.kge import kge_settings
from biomedkg.configs.kge_train import kge_train_settings


def train_on_batch(model, optimizer, loader, desc:str) -> float:
    total_loss = 0
    for idx, batch in enumerate(tqdm(loader, leave=True, desc=desc)):
        optimizer.zero_grad()
        batch.to(device)

        z = model.encode(batch.x, batch.edge_index, batch.edge_type)

        neg_edge_index = negative_sampling(batch.edge_index)

        pos_pred = model.decode(z, batch.edge_index, batch.edge_type)
        neg_pred = model.decode(z, neg_edge_index, batch.edge_type)
        pred = torch.cat([pos_pred, neg_pred])

        gt = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])

        cross_entropy_loss = F.binary_cross_entropy_with_logits(pred, gt)
        reg_loss = z.pow(2).mean() + model.decoder.rel_emb.pow(2).mean()
        loss = cross_entropy_loss + 1e-2 * reg_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.detach().cpu().detach()

        if idx % 100 == 0:
            print(f"\tBatch {idx}/{len(train_loader)}: {loss.detach().cpu().numpy():.3f}", flush=True)
    return total_loss


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO: Convert PrimeKG data from a Heterogeneous Graph to a Homogeneous Graph for easier implementation.
    data:torch_geometric.data.Data = None

    model = GAE(
        RGCN(
            in_channels = kge_settings.IN_DIMS, 
            num_hidden_layers = kge_settings.NUM_HIDDEN,
            hidden_channels = kge_settings.HIDDEN_DIM, 
            out_channels = kge_settings.OUT_DIM,
            num_relations = data.num_edge_types,
            drop_out = True),
        DistMult(data.num_edge_types, kge_settings.OUT_DIM),
    ).to(device)

    print(f"Total parameters: {parameters_count(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr = kge_train_settings.LEARNING_RATE)

    train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=kge_train_settings.NUM_VAL,
        num_test=kge_train_settings.NUM_TEST,
        neg_sampling_ratio=0.,
    )

    train_loader = GraphSAINTRandomWalkSampler(
        data = data,
        batch_size=kge_train_settings.BATCH_SIZE,
        walk_length=20,
        num_steps=kge_train_settings.STEP_PER_EPOCH,
        sample_coverage=100,
    )

    for epoch in range(kge_train_settings.EPOCHS):
        batch_loss = train_on_batch(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            desc=f"Epoch {epoch+1}/{kge_train_settings.EPOCHS}",
        )
        print(f"Epoch: {epoch + 1}, Loss: {batch_loss / len(train_loader):.4f}")
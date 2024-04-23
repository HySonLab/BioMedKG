import math
import torch
import torch.nn.functional as F


class Decoder(torch.nn.Module):
    def __init__(self, num_relations, hidden_channels):
        super().__init__()
        self.rel_emb = torch.nn.Embedding(num_embeddings=num_relations, embedding_dim=hidden_channels, sparse=True)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb)
    
    def forward(self, z, edge_index, edge_type):
        raise NotImplementedError
    

class TransE(Decoder):
    def __init__(self, num_relations, hidden_channels):
        super().__init__(num_relations, hidden_channels)
    
    def reset_parameters(self):
        bound = 6. / math.sqrt(self.hidden_channels)
        torch.nn.init.uniform_(self.rel_emb.weight, -bound, bound)
        F.normalize(self.rel_emb.weight.data, p=self.p_norm, dim=-1,
                    out=self.rel_emb.weight.data)
    
    def forward(self, z, edge_index, edge_type):
        head, tail = z[edge_index[0]], z[edge_index[1]]
        rel = self.rel_emb[edge_type]

        head = F.normalize(head, p=1.0, dim=-1)
        tail = F.normalize(tail, p=1.0, dim=-1)
        
        return -((head + rel) - tail).norm(p=1.0, dim=-1)


class DistMult(Decoder):
    def __init__(self, num_relations, hidden_channels):
        super().__init__(num_relations, hidden_channels)
    
    def forward(self, z, edge_index, edge_type):
        head, tail = z[edge_index[0]], z[edge_index[1]]
        rel = self.rel_emb[edge_type]

        return torch.sum(head * rel * tail, dim=1)
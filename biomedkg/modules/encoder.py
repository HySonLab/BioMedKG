import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, RGATConv


class RGCN(torch.nn.Module):
    def __init__(self, 
                 in_dims : int, 
                 hidden_dims : int, 
                 out_dims : int,
                 num_hidden_layers : int,
                 num_relations : int,
                 drop_out : bool = True
                 ):
        super().__init__()
        self.drop_out = drop_out

        self.graph_layers = []

        self.graph_layers.append(RGCNConv(in_channels=in_dims, out_channels=hidden_dims, num_relations=num_relations))
        for _ in range(num_hidden_layers):
            self.graph_layers.append(RGCNConv(in_channels=hidden_dims, out_channels=hidden_dims, num_relations=num_relations))
        self.graph_layers.append(RGCNConv(in_channels=hidden_dims, out_channels=out_dims, num_relations=num_relations))

        self.graph_layers = torch.nn.ModuleList(self.graph_layers)

        self.reset_parameters()

    def reset_parameters(self):
        for graph_layer in self.graph_layers:
            graph_layer.reset_parameters()

    def forward(self, x, edge_index, edge_type):
        for graph_layer in self.graph_layers[:-1]:
            x = F.relu(graph_layer(x, edge_index, edge_type))

            if self.drop_out:
                x = F.dropout(x, p=0.2, training=self.training)

        x = self.graph_layers[-1](x, edge_index, edge_type)
        
        return x
    

class RGAT(torch.nn.Module):
    def __init__(self, 
                 in_dims : int, 
                 hidden_dims : int, 
                 out_dims : int,
                 num_hidden_layers : int,
                 num_relations : int,
                 num_heads : int = 1,
                 drop_out : bool = True
                 ):
        super().__init__()
        self.drop_out = drop_out

        self.graph_layers = []

        self.graph_layers.append(RGATConv(in_channels=in_dims, out_channels=hidden_dims, num_relations=num_relations, heads=num_heads))
        for _ in range(num_hidden_layers):
            self.graph_layers.append(RGATConv(in_channels=hidden_dims, out_channels=hidden_dims, num_relations=num_relations, heads=num_heads))
        self.graph_layers.append(RGATConv(in_channels=hidden_dims, out_channels=out_dims, num_relations=num_relations, heads=num_heads))

        self.graph_layers = torch.nn.ModuleList(self.graph_layers)

        self.reset_parameters()

    def reset_parameters(self):
        for graph_layer in self.graph_layers:
            graph_layer.reset_parameters()

    def forward(self, x, edge_index, edge_type):
        for graph_layer in self.graph_layers[:-1]:
            x = F.relu(graph_layer(x, edge_index, edge_type))

            if self.drop_out:
                x = F.dropout(x, p=0.2, training=self.training)

        x = self.graph_layers[-1](x, edge_index, edge_type)
        
        return x
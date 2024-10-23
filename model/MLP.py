import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act=nn.ReLU, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = act()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.dropout(self.act(layer(x))) if i < self.num_layers - 1 else layer(x)
        return x
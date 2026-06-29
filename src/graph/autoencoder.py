import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import GAE


class GCNEncoder(torch.nn.Module):
    """
    Two-layer GCN encoder.
    """

    def __init__(self, in_channels, hidden_channels, latent_channels):
        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            latent_channels
        )

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)

        return x


from src.graph.config import GRAPH_CONFIG


def build_autoencoder():

    encoder = GCNEncoder(
        in_channels=GRAPH_CONFIG["in_channels"],
        hidden_channels=GRAPH_CONFIG["hidden_channels"],
        latent_channels=GRAPH_CONFIG["latent_channels"],
    )

    model = GAE(encoder)

    return model



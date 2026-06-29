import torch
from torch_geometric.data import Data
import numpy as np


def compute_edge_index(df, threshold=0.30):
    """
    Build graph edges using Pearson correlation.
    """

    corr = df.corr(method="pearson").abs()

    edges = []

    n = len(corr.columns)

    for i in range(n):
        for j in range(i + 1, n):

            if corr.iloc[i, j] > threshold:

                edges.append([i, j])
                edges.append([j, i])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return edge_index


def build_graph(sample, edge_index):
    """
    Convert one participant into a graph.
    """

    x = torch.tensor(
        sample.values,
        dtype=torch.float
    ).view(-1, 1)

    return Data(
        x=x,
        edge_index=edge_index
    )


def build_graph_dataset(df, threshold=0.30):

    edge_index = compute_edge_index(df, threshold)

    graphs = [
        build_graph(row, edge_index)
        for _, row in df.iterrows()
    ]

    return graphs, edge_index


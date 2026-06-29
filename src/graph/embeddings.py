import torch
import pandas as pd


def extract_embeddings(model, graphs):
    """
    Extract graph embeddings from a trained Graph Autoencoder.
    """

    device = next(model.parameters()).device

    model.eval()

    embeddings = []

    with torch.no_grad():

        for graph in graphs:

            graph = graph.to(device)

            z = model.encode(
                graph.x,
                graph.edge_index
            )

            # Graph embedding = mean of node embeddings
            graph_embedding = z.mean(dim=0)

            embeddings.append(
                graph_embedding.cpu().numpy()
            )

    embedding_df = pd.DataFrame(embeddings)

    embedding_df.columns = [
    f"GAE_Embedding_{i+1}"
    for i in range(embedding_df.shape[1])
    ]

    return embedding_df


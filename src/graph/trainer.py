import torch
from torch_geometric.loader import DataLoader

from src.graph.config import GRAPH_CONFIG


def train_autoencoder(model, graphs):
    """
    Train the Graph Autoencoder.

    Returns
    -------
    model : trained GAE
    loss_history : list
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)

    loader = DataLoader(
        graphs,
        batch_size=GRAPH_CONFIG["batch_size"],
        shuffle=GRAPH_CONFIG["shuffle"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=GRAPH_CONFIG["learning_rate"],
        weight_decay=GRAPH_CONFIG["weight_decay"],
    )

    model.train()

    loss_history = []

    for epoch in range(GRAPH_CONFIG["epochs"]):

        total_loss = 0

        for batch in loader:

            batch = batch.to(device)

            optimizer.zero_grad()

            z = model.encode(
                batch.x,
                batch.edge_index
            )

            loss = model.recon_loss(
                z,
                batch.edge_index
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        loss_history.append(avg_loss)

        print(
            f"Epoch {epoch+1:03d}/{GRAPH_CONFIG['epochs']} | "
            f"Loss = {avg_loss:.6f}"
        )

    return model, loss_history


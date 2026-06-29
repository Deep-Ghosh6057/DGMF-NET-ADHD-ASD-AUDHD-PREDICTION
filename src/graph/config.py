"""
Graph Autoencoder Configuration
"""

GRAPH_CONFIG = {
    # Model Architecture
    "in_channels": 1,
    "hidden_channels": 16,
    "latent_channels": 8,

    # Training Parameters
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "epochs": 100,

    # DataLoader
    "batch_size": 128,
    "shuffle": True,

    # Graph Construction
    "correlation_threshold": 0.30,
}




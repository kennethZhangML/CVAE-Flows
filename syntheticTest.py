import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from cvae import *

def create_synthetic_dihypergraph(num_nodes):
    """
    Create a synthetic dihypergraph adjacency matrix.
    Here, we just create a random binary matrix.
    """
    return torch.rand((num_nodes, num_nodes)) > 0.7

def create_synthetic_data(num_samples, input_dim):
    """
    Create synthetic data for the nodes in our dihypergraph.
    """
    return torch.rand((num_samples, input_dim))

def main():

    # Parameters
    input_dim = 784  # Example value; modify as needed
    latent_dim = 20  # Example value; modify as needed
    num_nodes = 100  # Number of nodes (participants) in our dihypergraph
    
    # Create synthetic dihypergraph and data
    adj_matrix = create_synthetic_dihypergraph(num_nodes)
    data = create_synthetic_data(num_nodes, input_dim)
    
    # Convert data to DataLoader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size = 32, shuffle=True)

    # Initialize and train the model
    model = StructuredCVAE(input_dim, latent_dim, adj_matrix)
    optimizer = optim.Adam(model.parameters())
    train(model, dataloader, optimizer, epochs=5)

    # Generate a new conversation node based on our model
    with torch.no_grad():
        sample = torch.randn(1, latent_dim)
        generated_data = model.decoder(sample)
        print("Generated Data:", generated_data)

if __name__ == "__main__":
    main()

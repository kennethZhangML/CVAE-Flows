import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from EDs import *

# GNN-based embedding for Structured C-VAE
class GNN(nn.Module):
    def __init__(self, num_nodes, latent_dim):
        super(GNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv1d(32, latent_dim, 3, padding=1)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x.view(x.size(0), -1)

# Structured C-VAE Model
class StructuredCVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, adj_matrix):
        super(StructuredCVAE, self).__init__()
        
        self.gnn = GNN(adj_matrix.size(0), latent_dim)
        self.embeddings = self.gnn(adj_matrix)
        
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
    
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, log_var = self.encoder(x)
        graph_reg = torch.norm(mean - self.embeddings)
        
        z = self.reparameterize(mean, log_var)
        x_recon = self.decoder(z)
        return x_recon, mean, log_var, graph_reg

def loss_function(x_recon, x, mean, log_var, graph_reg, reg_weight = 0.1):
    recon_loss = nn.MSELoss()(x_recon, x)
    kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return recon_loss + kl_div + reg_weight * graph_reg

def train(model, dataloader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for (x,) in dataloader:
            x_recon, mean, log_var, graph_reg = model(x)
            loss = loss_function(x_recon, x, mean, log_var, graph_reg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item()}")

if __name__ == "__main__":
    adj_matrix = torch.tensor([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ], dtype = torch.float32)

    data_samples = torch.rand(100, 784)
    dataset = TensorDataset(data_samples)
    train_dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

    input_dim = 784
    latent_dim = 20

    model = StructuredCVAE(input_dim, latent_dim, adj_matrix)
    optimizer = optim.Adam(model.parameters())

    train(model, train_dataloader, optimizer, epochs = 10)
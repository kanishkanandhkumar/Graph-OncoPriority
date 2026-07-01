import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import os

class GenePrioritizerGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels // 2, heads=2)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

def train():
    # Load the graph
    data = torch.load("data/processed/graph.pt")
    model = GenePrioritizerGNN(in_channels=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    print("--- Training GNN ---")
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
            
    torch.save(model.state_dict(), "results/best_model.pt")
    print("--- Training Complete. Model saved to results/best_model.pt ---")

if __name__ == "__main__":
    train()
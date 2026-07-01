import torch
import pickle
import numpy as np

def analyze_importance(node_idx):
    # Load your edge mask and original edge indices
    edge_mask = torch.load("results/edge_mask.pt")
    data = torch.load("data/processed/graph.pt", weights_only=False)
    with open("data/processed/gene_names.pkl", "rb") as f:
        idx_to_protein = pickle.load(f)

    # Get edges connected to your node
    edge_index = data.edge_index
    # Find indices where this node is involved
    mask = (edge_index[0] == node_idx) | (edge_index[1] == node_idx)
    relevant_edges = edge_index[:, mask]
    relevant_weights = edge_mask[mask]

    # Sort by importance
    top_indices = torch.argsort(relevant_weights, descending=True)
    
    print(f"Top drivers for {idx_to_protein[node_idx]}:")
    for i in range(min(5, len(top_indices))):
        idx = top_indices[i]
        neighbor = relevant_edges[:, idx]
        target = neighbor[1] if neighbor[0] == node_idx else neighbor[0]
        print(f" - Connected to {idx_to_protein[target.item()]} (Importance: {relevant_weights[idx]:.4f})")

if __name__ == "__main__":
    analyze_importance(node_idx=0)
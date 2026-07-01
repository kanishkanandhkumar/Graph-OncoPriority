import torch
from torch_geometric.nn import GNNExplainer
from src.model import GenePrioritizerGNN
import pickle

def explain_gene(target_gene_idx=0):
    # Load data and model
    data = torch.load("data/processed/graph.pt", weights_only=False)
    model = GenePrioritizerGNN(in_channels=5)
    model.load_state_dict(torch.load("results/best_model.pt", weights_only=True))
    model.eval()

    # Initialize Explainer
    # epochs=200 is sufficient for a quick visualization
    explainer = GNNExplainer(model, epochs=200, return_type='log_prob')

    # Explain the prediction for a specific node
    node_idx = target_gene_idx 
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, data.x, data.edge_index)

    # Save results
    torch.save(edge_mask, "results/edge_mask.pt")
    print(f"Explanation saved! Important edges identified for node {node_idx}.")

if __name__ == "__main__":
    # Explain the first high-confidence gene
    explain_gene(target_gene_idx=0)
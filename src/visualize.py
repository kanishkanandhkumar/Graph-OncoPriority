import torch
import torch.nn.functional as F
import pandas as pd
import pickle
from src.model import GenePrioritizerGNN

def visualize():
    # Set weights_only=False to allow loading the graph data structure
    data = torch.load("data/processed/graph.pt", weights_only=False)
    
    model = GenePrioritizerGNN(in_channels=5)
    model.load_state_dict(torch.load("results/best_model.pt", weights_only=True))
    model.eval()
    
    with open("data/processed/gene_names.pkl", "rb") as f:
        idx_to_protein = pickle.load(f)
        
    out = F.softmax(model(data.x, data.edge_index), dim=1)
    
    results = pd.DataFrame({
        'Gene': [idx_to_protein[i] for i in range(len(idx_to_protein))],
        'Score': out[:, 1].detach().numpy()
    })
    
    top_50 = results.sort_values(by='Score', ascending=False).head(50)
    top_50.to_csv("results/tables/top_predictions.csv", index=False)
    print("Top 50 predictions saved to results/tables/top_predictions.csv")

if __name__ == "__main__":
    visualize()
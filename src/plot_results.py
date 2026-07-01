import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from src.model import GenePrioritizerGNN

def plot_distributions():
    # 1. Load data and model
    data = torch.load("data/processed/graph.pt", weights_only=False)
    model = GenePrioritizerGNN(in_channels=5)
    model.load_state_dict(torch.load("results/best_model.pt", weights_only=True))
    model.eval()
    
    # 2. Get predictions for ALL genes
    with torch.no_grad():
        out = F.softmax(model(data.x, data.edge_index), dim=1)
        all_scores = out[:, 1].numpy()
    
    # 3. Separate scores by label (1 = cancer, 0 = non-cancer)
    cancer_scores = all_scores[data.y.numpy() == 1]
    non_cancer_scores = all_scores[data.y.numpy() == 0]
    
    # 4. Plot
    plt.figure(figsize=(10, 6))
    plt.hist(non_cancer_scores, bins=50, alpha=0.5, label='Non-Cancer (Background)', color='blue')
    plt.hist(cancer_scores, bins=50, alpha=0.7, label='Known Cancer Hubs', color='red')
    
    plt.title("Model Prediction Distribution: Cancer vs. Non-Cancer Genes")
    plt.xlabel("Model Probability Score")
    plt.ylabel("Number of Genes")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Save
    plt.savefig("results/figures/prediction_distribution.png")
    print("Plot saved to results/figures/prediction_distribution.png")

if __name__ == "__main__":
    plot_distributions()
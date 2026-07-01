import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
from src.model import GenePrioritizerGNN

def evaluate():
    # Set weights_only=False to allow loading the graph data structure
    data = torch.load("data/processed/graph.pt", weights_only=False)
    
    model = GenePrioritizerGNN(in_channels=5)
    model.load_state_dict(torch.load("results/best_model.pt", weights_only=True))
    model.eval()
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
    test_mask = data.test_mask
    auc = roc_auc_score(data.y[test_mask].cpu(), out[test_mask, 1].cpu())
    f1 = f1_score(data.y[test_mask].cpu(), pred[test_mask].cpu())
    
    print("\n--- Evaluation Results ---")
    print(f"Test ROC-AUC: {auc:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print("--------------------------\n")

if __name__ == "__main__":
    evaluate()
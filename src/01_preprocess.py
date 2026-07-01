import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import pickle

def main():
    print("--- Starting Data Preprocessing (Graph Construction) ---\n")
    
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    string_file = os.path.join(raw_dir, "string_interactions.tsv")
    
    # 1. Load and Filter Protein Interactions
    print("1. Loading STRING database...")
    # STRING has columns: protein1, protein2, neighborhood, fusion, cooccurence, coexpression, experimental, database, textmining, combined_score
    df = pd.read_csv(string_file, sep=' ')
    
    print(f"Original interactions: {len(df):,}")
    
    # Filter for high confidence interactions (>700 out of 1000)
    df = df[df['combined_score'] > 700].copy()
    print(f"High-confidence interactions (>0.7): {len(df):,}")
    
    # 2. Create Node Mappings (Strings to Integers)
    print("\n2. Mapping proteins to graph nodes...")
    unique_proteins = pd.unique(df[['protein1', 'protein2']].values.ravel('K'))
    num_nodes = len(unique_proteins)
    
    protein_to_idx = {prot: i for i, prot in enumerate(unique_proteins)}
    idx_to_protein = {i: prot for i, prot in enumerate(unique_proteins)}
    
    print(f"Total unique genes/proteins (Nodes): {num_nodes:,}")
    
    # 3. Build Edge Index for PyTorch Geometric
    print("\n3. Building Edge Index...")
    # Map the string IDs to our new integer IDs
    src = [protein_to_idx[prot] for prot in df['protein1']]
    dst = [protein_to_idx[prot] for prot in df['protein2']]
    
    # PyG expects edge_index of shape [2, num_edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # 4. Generate Node Features (5 Dimensions as per architecture)
    print("\n4. Generating Node Features...")
    # In a full production build, you'd pull GC content, length, etc.
    # Here we generate a structural feature (node degree) + biological proxies
    node_degrees = torch.bincount(edge_index[0], minlength=num_nodes).float()
    node_degrees = (node_degrees - node_degrees.mean()) / node_degrees.std() # Normalize
    
    # Create 5 feature columns (Degree + 4 random normal features simulating expression/length/etc)
    x = torch.zeros((num_nodes, 5), dtype=torch.float)
    x[:, 0] = node_degrees
    x[:, 1:] = torch.randn((num_nodes, 4))
    
    # 5. Create Labels (Cancer vs Non-Cancer)
    print("\n5. Assigning Labels & Splits...")
    y = torch.zeros(num_nodes, dtype=torch.long)
    
    # Simulate known cancer genes: For this starter, we assign the top 1.5% 
    # most connected nodes as our known "Cancer Genes" (hubs)
    num_cancer_genes = int(num_nodes * 0.015)
    top_degree_indices = torch.argsort(node_degrees, descending=True)[:num_cancer_genes]
    y[top_degree_indices] = 1
    
    print(f"Known Cancer Genes (Class 1): {num_cancer_genes:,}")
    print(f"Unknown/Non-Cancer (Class 0): {num_nodes - num_cancer_genes:,}")
    
    # 6. Create Train/Val/Test Masks (70% / 15% / 15%)
    indices = torch.randperm(num_nodes)
    train_end = int(num_nodes * 0.70)
    val_end = int(num_nodes * 0.85)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True
    
    # 7. Construct PyG Data Object
    print("\n7. Saving PyTorch Graph...")
    graph_data = Data(x=x, edge_index=edge_index, y=y, 
                      train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    # Save the graph
    torch.save(graph_data, os.path.join(processed_dir, "graph.pt"))
    
    # Save the dictionaries so we know which ID belongs to which gene later
    with open(os.path.join(processed_dir, "gene_names.pkl"), 'wb') as f:
        pickle.dump(idx_to_protein, f)
        
    print(f"Graph saved to: {processed_dir}/graph.pt")
    print("--- Preprocessing Complete ---")

if __name__ == "__main__":
    main()
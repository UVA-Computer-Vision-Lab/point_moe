#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import os
import argparse
from pathlib import Path

def extract_layer_name(col):
    """Extract layer name from column name."""
    return col

def plot_dataset_heatmap(dataset_name, data, save_dir):
    """Plot heatmaps for token distribution and gate scores for a dataset."""
    ratio_data = data[f'{dataset_name}_ratio']
    gate_scores = data.get(f'{dataset_name}_gate_scores', {})
    
    # Convert to DataFrame for easier plotting
    df_ratio = pd.DataFrame(ratio_data)
    df_ratio.columns = [extract_layer_name(col) for col in df_ratio.columns]
    
    # Create figure for token distribution
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_ratio, 
                cmap='YlOrRd',
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Token Distribution Ratio'},
                square=True)
    
    plt.title(f'{dataset_name} Token Distribution Ratios')
    plt.xlabel('Layer')
    plt.ylabel('Expert')
    
    # Save in the new directory
    plt.savefig(save_dir / f'{dataset_name}_token_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot gate scores if they exist
    if gate_scores:
        df_scores = pd.DataFrame(gate_scores)
        df_scores.columns = [extract_layer_name(col) for col in df_scores.columns]
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(df_scores, 
                    cmap='YlOrRd',
                    annot=True,
                    fmt='.2f',
                    cbar_kws={'label': 'Gate Scores'},
                    square=True)
        
        plt.title(f'{dataset_name} Gate Scores')
        plt.xlabel('Layer')
        plt.ylabel('Expert')
        
        plt.savefig(save_dir / f'{dataset_name}_gate_scores.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize MoE token distribution and gate scores.')
    parser.add_argument('data_path', type=str, help='Path to the moe_token_distribution.pkl file')
    args = parser.parse_args()

    # Load the data
    data = pickle.load(open(args.data_path, 'rb'))

    # Get all datasets
    datasets = [key for key in data.keys() if not key.endswith('_ratio') and not "scores" in key]

    pickle_path = Path(args.data_path)
    base_name = pickle_path.stem
    save_dir = pickle_path.parent / base_name

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Plot for each dataset
    for dataset in datasets:
        plot_dataset_heatmap(dataset, data, save_dir)

    print(f"Plots saved in: {save_dir}")

if __name__ == "__main__":
    main()
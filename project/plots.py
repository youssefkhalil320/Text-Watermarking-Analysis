import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create the plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Load the datasets
datasets = {
    'xsum-watermarked-flan-t5-small': pd.read_csv('data/xsum-watermarked-flan-t5-small_dataset.csv'),
    'xsum-watermarked-t5-small': pd.read_csv('data/xsum-watermarked-t5-small_dataset.csv'),
    'cnn_dailymail-watermarked-flan-t5-small': pd.read_csv('data/cnn_dailymail-watermarked-flan-t5-small_dataset.csv'),
    'cnn_dailymail-watermarked-t5-small': pd.read_csv('data/cnn_dailymail-watermarked-t5-small_dataset.csv')
}

# Define a function to plot the scores and save them
def plot_scores(dataset_name, df):
    metrics = ['bleu_score', 'rouge1_score', 'rouge2_score', 'rougeL_score', 'meteor_score']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(df[f'summary_{metric}'], label='Summary', shade=True)
        sns.kdeplot(df[f'watermarked_{metric}'], label='Watermarked Summary', shade=True)
        plt.title(f'{metric.replace("_", " ").title()} Distribution - {dataset_name}')
        plt.xlabel(f'{metric.replace("_", " ").title()}')
        plt.ylabel('Density')
        plt.legend()
        
        # Save the plot
        filename = f'plots/{dataset_name}_{metric}.png'
        plt.savefig(filename)
        plt.close()  # Close the plot to avoid displaying it

# Plot and save scores for each dataset
for dataset_name, df in datasets.items():
    plot_scores(dataset_name, df)

# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Create the plots directory if it doesn't exist
# os.makedirs('plots', exist_ok=True)

# # Load the datasets
# datasets = {
#     'xsum-watermarked-flan-t5-small': pd.read_csv('data/xsum-watermarked-flan-t5-small_dataset.csv'),
#     'xsum-watermarked-t5-small': pd.read_csv('data/xsum-watermarked-t5-small_dataset.csv'),
#     'cnn_dailymail-watermarked-flan-t5-small': pd.read_csv('data/cnn_dailymail-watermarked-flan-t5-small_dataset.csv'),
#     'cnn_dailymail-watermarked-t5-small': pd.read_csv('data/cnn_dailymail-watermarked-t5-small_dataset.csv')
# }

# # Define a function to plot the scores and save them
# def plot_scores(dataset_name, df):
#     metrics = ['bleu_score', 'rouge1_score', 'rouge2_score', 'rougeL_score', 'meteor_score']
    
#     for metric in metrics:
#         plt.figure(figsize=(10, 6))
#         sns.kdeplot(df[f'summary_{metric}'], label='Summary', shade=True)
#         sns.kdeplot(df[f'watermarked_{metric}'], label='Watermarked Summary', shade=True)
#         plt.title(f'{metric.replace("_", " ").title()} Distribution - {dataset_name}')
#         plt.xlabel(f'{metric.replace("_", " ").title()}')
#         plt.ylabel('Density')
#         plt.legend()
        
#         # Save the plot
#         filename = f'plots/{dataset_name}_{metric}.png'
#         plt.savefig(filename)
#         plt.close()  # Close the plot to avoid displaying it

# # Plot and save scores for each dataset
# for dataset_name, df in datasets.items():
#     plot_scores(dataset_name, df)

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


# Create the CSV file to store average scores
averages_file = 'averages.csv'

# Define a function to plot the scores, save them, and write averages to a CSV file
def plot_scores_and_write_averages(dataset_name, df, averages_df):
    metrics = ['bleu_score', 'rouge1_score', 'rouge2_score', 'rougeL_score', 'meteor_score']
    
    # Calculate averages and prepare data for CSV
    averages = {'Dataset': dataset_name}
    for metric in metrics:
        summary_avg = df[f'summary_{metric}'].mean()
        watermarked_avg = df[f'watermarked_{metric}'].mean()
        averages[f'{metric}_summary'] = summary_avg
        averages[f'{metric}_watermarked'] = watermarked_avg
        print(f'  {metric.replace("_", " ").title()} - Summary: {summary_avg:.4f}, Watermarked: {watermarked_avg:.4f}')
    
    # Convert the averages dictionary to a DataFrame and concatenate it
    averages_df = pd.concat([averages_df, pd.DataFrame([averages])], ignore_index=True)

    # Plot the score distributions
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

    return averages_df

# Initialize an empty DataFrame to store averages
averages_df = pd.DataFrame()

# Plot, save scores, and write averages for each dataset
for dataset_name, df in datasets.items():
    averages_df = plot_scores_and_write_averages(dataset_name, df, averages_df)

# Save the averages DataFrame to a CSV file
averages_df.to_csv(averages_file, index=False)

print(f'\nAverages have been written to {averages_file}.')

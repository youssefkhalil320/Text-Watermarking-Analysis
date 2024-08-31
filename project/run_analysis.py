import nltk
import pandas as pd
from datasets import load_dataset
from modules.bleu_score import calculate_bleu
from modules.rouge_score import add_rouge_scores
from modules.meteor_scores import add_meteor_scores

# Download necessary resources for METEOR and tokenization
nltk.download('wordnet')
nltk.download('punkt')

#xsum-watermarked-flan-t5-small
#xsum-watermarked-t5-small
#
#

# Load your dataset
account_name = "youssefkhalil320"
dataset_name = "cnn_dailymail-watermarked-t5-small"
full_dataset_name = f"{account_name}/{dataset_name}"

dataset = load_dataset(full_dataset_name)

# Calculate BLEU scores
dataset = dataset.map(lambda x: {'summary_bleu_score': calculate_bleu(x['document'], x['summary'])})
dataset = dataset.map(lambda x: {'watermarked_bleu_score': calculate_bleu(x['document'], x['watermarked_summary'])})

# Add ROUGE and METEOR scores
dataset = dataset.map(add_rouge_scores)
dataset = dataset.map(add_meteor_scores)

# Convert the dataset to a Pandas DataFrame
df = pd.DataFrame(dataset['train'])

# Save the DataFrame to a CSV file
csv_file_path = f"data/{dataset_name}_dataset.csv"
df.to_csv(csv_file_path, index=False)

print(f"Dataset saved to {csv_file_path}")

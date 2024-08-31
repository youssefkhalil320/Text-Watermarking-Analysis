import nltk
from datasets import load_dataset
from modules.bleu_score import calculate_bleu
from modules.rouge_score import add_rouge_scores
from modules.meteor_scores import add_meteor_scores

nltk.download('wordnet')
nltk.download('punkt')

account_name = "youssefkhalil320"
dataset_name = "xsum-watermarked-flan-t5-small"
full_dataset_name = f"{account_name}/{dataset_name}"

dataset = load_dataset(full_dataset_name)

dataset = dataset.map(lambda x: {'summary_bleu_score': calculate_bleu(x['document'], x['summary'])})
dataset = dataset.map(lambda x: {'watermarked_bleu_score': calculate_bleu(x['document'], x['watermarked_summary'])})
dataset = dataset.map(add_rouge_scores)
dataset = dataset.map(add_meteor_scores)


print(dataset)
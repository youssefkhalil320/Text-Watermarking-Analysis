import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize


# Function to calculate METEOR score
def calculate_meteor(reference, summary):
    reference_tokens = word_tokenize(reference)
    summary_tokens = word_tokenize(summary)
    return meteor_score([reference_tokens], summary_tokens)

# Function to add METEOR scores to the dataset
def add_meteor_scores(example):
    return {
        'summary_meteor_score': calculate_meteor(example['document'], example['summary']),
        'watermarked_meteor_score': calculate_meteor(example['document'], example['watermarked_summary'])
    }
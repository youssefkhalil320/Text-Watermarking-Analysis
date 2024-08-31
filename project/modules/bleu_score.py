from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu(reference, summary):
    return sentence_bleu([reference.split()], summary.split())


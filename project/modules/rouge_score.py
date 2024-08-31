from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_rouge(reference, summary):
    scores = scorer.score(reference, summary)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure,
    }

def add_rouge_scores(example):
    rouge_summary_scores = calculate_rouge(example['document'], example['summary'])
    rouge_watermarked_scores = calculate_rouge(example['document'], example['watermarked_summary'])

    return {
        'summary_rouge1_score': rouge_summary_scores['rouge1'],
        'summary_rouge2_score': rouge_summary_scores['rouge2'],
        'summary_rougeL_score': rouge_summary_scores['rougeL'],
        'watermarked_rouge1_score': rouge_watermarked_scores['rouge1'],
        'watermarked_rouge2_score': rouge_watermarked_scores['rouge2'],
        'watermarked_rougeL_score': rouge_watermarked_scores['rougeL']
    }


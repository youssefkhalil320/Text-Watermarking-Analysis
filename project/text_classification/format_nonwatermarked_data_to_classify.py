import pandas as pd

# Load the datasets
flan_df = pd.read_csv('../data/cnn_dailymail_flan_t5_dataset.csv')
t5_df = pd.read_csv('../data/cnn_dailymail_t5_dataset.csv')



# Create the second combined dataset with watermarked summaries
watermarked_summary_combined_df = pd.DataFrame({
    'text': pd.concat([t5_df['t5_summary'], flan_df['flan_t5_summary']], ignore_index=True),
    'label': [0] * len(t5_df) + [1] * len(flan_df)
})

print(watermarked_summary_combined_df.shape)


watermarked_summary_combined_df = watermarked_summary_combined_df.sample(frac=1).reset_index(drop=True)

print(watermarked_summary_combined_df.shape)

# Save the new datasets
watermarked_summary_combined_df.to_csv('new_data/cnn_dailymail_summary_unwatermarked_dataset.csv', index=False)

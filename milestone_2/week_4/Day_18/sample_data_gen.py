import pandas as pd
# Load the full dataset (replace with your actual full data file path)
full_data_path = "C:/Users/win10/Desktop/US_Accidents_March23.csv"
df = pd.read_csv(full_data_path)

# Sample 1 million rows (or all if dataset smaller)
sample_size = min(1_000_000, len(df))
df_sampled = df.sample(n=sample_size, random_state=42)  # random_state for reproducibility

# Optional: reset index
df_sampled.reset_index(drop=True, inplace=True)

# Save sampled data
output_path = "US_Accidents_March23_sampled_1M.csv"
df_sampled.to_csv(output_path, index=False)

print(f"Sampled {sample_size} rows saved to {output_path}")

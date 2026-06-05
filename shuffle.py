import pandas as pd

df = pd.read_csv(r"C:\Users\Rifah Nanziba\Downloads\VQA-With-Multimodal-Transformers\VQA dataset code run.csv")

# Fix 1: Drop the empty extra column
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Fix 2: Drop any blank rows
df = df.dropna(subset=['image_id', 'question', 'answer'])
df = df[df['image_id'].str.strip() != '']

print("Clean total rows:", len(df))

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split 80/20
train = df.iloc[:3200]
eval  = df.iloc[3200:]

# Save
train.to_csv(r"C:\Users\Rifah Nanziba\Downloads\VQA-With-Multimodal-Transformers\dataset\data_train.csv", index=False)
eval.to_csv(r"C:\Users\Rifah Nanziba\Downloads\VQA-With-Multimodal-Transformers\dataset\data_eval.csv", index=False)

print("Done! Train:", len(train), "| Eval:", len(eval))
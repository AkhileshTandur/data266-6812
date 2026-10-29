
from datasets import load_dataset
import pandas as pd

print("Downloading Sentiment140 from Hugging Face...")

dataset = load_dataset(
    "bdanko/sentiment140",
    split="train"
)

df = dataset.to_pandas()

print("Original rows:", len(df))
print("Columns:", df.columns.tolist())

df["label"] = df["sentiment"].map({
    0: 0,
    4: 1
})

if df["label"].isna().any():
    raise ValueError("Unexpected sentiment labels")

df["label"] = df["label"].astype(int)

df[["text", "label"]].to_csv(
    "twitter.csv",
    index=False
)

print("Download completed!")
print("Saved: twitter.csv")
print("Total rows:", len(df))
print(df["label"].value_counts())

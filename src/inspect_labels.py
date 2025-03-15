import pandas as pd

train_df = pd.read_csv("data/processed/train_data.csv", parse_dates=["timestamp"])
test_df = pd.read_csv("data/processed/test_data.csv", parse_dates=["timestamp"])

print("Train Data Label Distribution:")
print(train_df["label"].value_counts())

print("Test Data Label Distribution:")
print(test_df["label"].value_counts())

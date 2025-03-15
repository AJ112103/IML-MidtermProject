import pandas as pd
import numpy as np
import os

def generate_labels(df, horizon=3, threshold=0.00005):
    df = df.sort_values(by=["symbol", "timestamp"]).copy()
    def compute_label(group):
        group["future_mid"] = group["mid_price"].shift(-horizon)
        group["pct_change"] = (group["future_mid"] - group["mid_price"]) / group["mid_price"]
        group["label"] = np.where(group["pct_change"] > threshold, 1,
                          np.where(group["pct_change"] < -threshold, -1, 0))
        return group
    df = df.groupby("symbol", group_keys=False).apply(compute_label)
    df = df.dropna(subset=["label"])
    return df

def train_test_split(df, train_frac=0.7, random_split=True, random_seed=42):
    if random_split:
        df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        train_size = int(len(df_shuffled) * train_frac)
        train_df = df_shuffled.iloc[:train_size].copy()
        test_df = df_shuffled.iloc[train_size:].copy()
    else:
        df = df.sort_values(by="timestamp")
        train_size = int(len(df) * train_frac)
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()
    return train_df, test_df

if __name__ == "__main__":
    df_features = pd.read_csv("data/processed/features_orderbooks.csv", parse_dates=["timestamp"])
    df_labeled = generate_labels(df_features, horizon=3, threshold=0.00005)
    train_df, test_df = train_test_split(df_labeled, train_frac=0.7, random_split=True)
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train_data.csv", index=False)
    test_df.to_csv("data/processed/test_data.csv", index=False)
    print("Files saved as 'train_data.csv' and 'test_data.csv'.")

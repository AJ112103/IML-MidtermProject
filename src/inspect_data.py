import pandas as pd

train_df = pd.read_csv("data/processed/train_data.csv", parse_dates=["timestamp"])
print("Train Data Label Distribution:")
print(train_df["label"].value_counts())

features_df = pd.read_csv("data/processed/features_orderbooks.csv", parse_dates=["timestamp"])
features = ["mid_price", "spread", "volume_imbalance", "rolling_volatility"]
print("\nFeature Descriptive Statistics:")
print(features_df[features].describe())

def compute_pct_stats(df, horizon=3):
    df = df.sort_values(by=["symbol", "timestamp"]).copy()
    df["future_mid"] = df["mid_price"].shift(-horizon)
    df["pct_change"] = (df["future_mid"] - df["mid_price"]) / df["mid_price"]
    return df

df_with_pct = compute_pct_stats(features_df, horizon=3)
print("\nPercentage Change Statistics:")
print(df_with_pct["pct_change"].describe())

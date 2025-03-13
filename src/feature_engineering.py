import pandas as pd
import numpy as np

def compute_features(df, window =10, epsilon=1e-8):

    df = df.sort_values(by=["symbol", "timestamp"]).copy()

    df["mid_price"] = (df["best_bid_price"] + df["best_ask_price"]) / 2
    df["spread"] = df["best_ask_price"] - df["best_bid_price"]

    df["volume_imbalance"] = (df["best_bid_qty"] - df["best_ask_qty"]) / (df["best_bid_qty"] + df["best_ask_qty"] + epsilon)

    df["rolling_volatility"] = df.groupby("symbol")["mid_price"].transform(lambda x: x.rolling(window, min_periods=1).std())

    return df

if __name__ == "__main__":

    df_clean = pd.read_csv("data/processed/clean_orderbooks.csv", parse_dates=["timestamp"])

    df_features = compute_features(df_clean, window=10)

    output_path = "data/processed/features_orderbooks.csv"
    df_features.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")
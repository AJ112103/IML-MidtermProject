import os
import json
import pandas as pd
import numpy as np

def parse_raw_orderbook(file_path):

    with open(file_path, "r") as f:
        data = json.load(f)

    if "bids" not in data or len(data["bids"]) == 0:
        return None
    if "asks" not in data or len(data["asks"]) == 0:
        return None
    
    best_bid_price, best_bid_qty = data["bids"][0]
    best_ask_price, best_ask_qty = data["asks"][0]

    filename = os.path.basename(file_path)
    parts = filename.split('_')
    if len(parts) >= 3:
        symbol = parts[0]
        timestamp_str = parts[2].replace(".json", "")
        try:
            timestamp = pd.to_datetime(int(timestamp_str), unit="ms")
        except Exception as e:
            timestamp = None
    else:
        symbol = None
        timestamp = None
    
    record = {
        "symbol": symbol,
        "timestamp": timestamp,
        "best_bid_price": float(best_bid_price),
        "best_bid_qty": float(best_bid_qty),
        "best_ask_price": float(best_ask_price),
        "best_ask_qty": float(best_ask_qty),
    }

    return pd.DataFrame([record])

def clean_all_raw(folder="data/raw"):
    all_dfs = []
    for f in os.listdir(folder):
        if f.endswith(".json"):
            full_path = os.path.join(folder, f)
            df = parse_raw_orderbook(full_path)
            if df is not None:
                all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()
    
    final_df = pd.concat(all_dfs, ignore_index=True)
    return final_df

if __name__ == "__main__":
    df_clean = clean_all_raw()
    print(df_clean.head())
    processed_folder = "data/processed"
    os.makedirs(processed_folder, exist_ok=True)
    output_file = os.path.join(processed_folder, "clean_orderbooks.csv")
    df_clean.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
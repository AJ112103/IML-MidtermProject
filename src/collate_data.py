import os
import json
import pandas as pd

def parse_raw_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    if "bids" not in data or "asks" not in data or not data["bids"] or not data["asks"]:
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
        except:
            timestamp = None
    else:
        symbol = None
        timestamp = None
    return {
        "symbol": symbol,
        "timestamp": timestamp,
        "best_bid_price": float(best_bid_price),
        "best_bid_qty": float(best_bid_qty),
        "best_ask_price": float(best_ask_price),
        "best_ask_qty": float(best_ask_qty)
    }

def collate_data(raw_folder="data/raw", output_file="data/processed/collected_orderbooks.csv"):
    records = []
    for filename in os.listdir(raw_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(raw_folder, filename)
            record = parse_raw_file(file_path)
            if record:
                records.append(record)
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    print(f"Collected data saved to {output_file}")

if __name__ == "__main__":
    collate_data()

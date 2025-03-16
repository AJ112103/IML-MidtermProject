import pandas as pd
df_train = pd.read_csv("data/processed/train_data.csv", parse_dates=["timestamp"])
print("Unique symbols in train_data.csv:")
print(df_train["symbol"].unique())

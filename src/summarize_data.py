import pandas as pd

def main():
    df_train = pd.read_csv("data/processed/train_data.csv", parse_dates=["timestamp"])

    print("First 20 rows of train_data:")
    print(df_train.head(20))

    print("\nDataFrame Describe:")
    print(df_train.describe())

    print("\nUnique Symbols in train_data:")
    print(df_train["symbol"].unique())

    print("\nLabel Distribution in train_data:")
    print(df_train["label"].value_counts())

    print("\nLabel distribution by symbol:")
    print(df_train.groupby("symbol")["label"].value_counts())

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np

def add_polynomial_features(df, features):
    df_poly = df.copy()
    for feature in features:
        df_poly[f'{feature}_sq'] = df_poly[feature] ** 2
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            df_poly[f'{features[i]}_x_{features[j]}'] = df_poly[features[i]] * df_poly[features[j]]
    return df_poly

if __name__ == '__main__':
    df = pd.read_csv("data/processed/features_orderbooks.csv", parse_dates=["timestamp"])
    features = ["mid_price", "spread", "volume_imbalance", "rolling_volatility"]
    df_poly = add_polynomial_features(df, features)
    output_path = "data/processed/enhanced_features_orderbooks.csv"
    df_poly.to_csv(output_path, index=False)
    print(f"Enhanced features saved to {output_path}")

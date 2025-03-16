import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from logistic_regression import predict as predict_logreg
from kernel_logistic_regression import predict as predict_kernel, random_fourier_features
from deep_mlp import predict_deep_mlp

def load_logistic_model():
    W = np.load("data/processed/model_weights.npy")
    mu = np.load("data/processed/feature_mu.npy")
    sigma = np.load("data/processed/feature_sigma.npy")
    return W, mu, sigma

def load_kernel_model():
    W = np.load("data/processed/kernel_model_weights.npy")
    mu = np.load("data/processed/kernel_feature_mu.npy")
    sigma = np.load("data/processed/kernel_feature_sigma.npy")
    return W, mu, sigma

def load_deep_mlp_model():
    params = {}
    params["W1"] = np.load("data/processed/deep_mlp_W1.npy")
    params["b1"] = np.load("data/processed/deep_mlp_b1.npy")
    params["W2"] = np.load("data/processed/deep_mlp_W2.npy")
    params["b2"] = np.load("data/processed/deep_mlp_b2.npy")
    params["W3"] = np.load("data/processed/deep_mlp_W3.npy")
    params["b3"] = np.load("data/processed/deep_mlp_b3.npy")
    params["W4"] = np.load("data/processed/deep_mlp_W4.npy")
    params["b4"] = np.load("data/processed/deep_mlp_b4.npy")
    params["W5"] = np.load("data/processed/deep_mlp_W5.npy")
    params["b5"] = np.load("data/processed/deep_mlp_b5.npy")

    mu = np.load("data/processed/deep_mlp_feature_mu.npy")
    sigma = np.load("data/processed/deep_mlp_feature_sigma.npy")
    return params, mu, sigma

def main():
    df = pd.read_csv("data/processed/test_data.csv", parse_dates=["timestamp"])
    features = ["mid_price","spread","volume_imbalance","rolling_volatility"]
    df = df.dropna(subset=features+["label"])

    # Sample a small subset
    df_small = df.sample(n=20, random_state=42).copy().reset_index(drop=True)
    df_small.sort_values("timestamp", inplace=True)

    X_all = df_small[features].values
    y_true = df_small["label"].values

    # 1) Logistic
    W_log, mu_log, sigma_log = load_logistic_model()
    X_norm_log = (X_all - mu_log)/sigma_log
    preds_log = predict_logreg(X_norm_log, W_log)

    # 2) Kernel
    W_kernel, mu_kernel, sigma_kernel = load_kernel_model()
    from kernel_logistic_regression import random_fourier_features
    X_norm_k = (X_all - mu_kernel)/sigma_kernel
    Z_k = random_fourier_features(X_norm_k, D=500, gamma=0.001)
    preds_kernel = predict_kernel(Z_k, W_kernel)

    # 3) Deep MLP
    params_deep, mu_deep, sigma_deep = load_deep_mlp_model()
    X_norm_deep = (X_all - mu_deep)/sigma_deep
    preds_deep = predict_deep_mlp(X_norm_deep, params_deep)

    print("timestamp          symbol  true   log_reg  kernel   deep_mlp")
    print("------------------------------------------------------------")
    for i in range(len(df_small)):
        ts = df_small.loc[i, "timestamp"]
        sym = df_small.loc[i, "symbol"]
        print(f"{ts} {sym:8s} {y_true[i]:>5d}  {preds_log[i]:>7d}  {preds_kernel[i]:>7d}  {preds_deep[i]:>7d}")

if __name__ == "__main__":
    main()

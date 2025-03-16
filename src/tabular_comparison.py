import pandas as pd
import numpy as np
from logistic_regression import predict as predict_logreg
from kernel_logistic_regression import (predict as predict_kernel,
                                        random_fourier_features)
from deep_mlp import predict_deep_mlp

def load_kernel_model():
    W_kernel = np.load("data/processed/kernel_model_weights.npy")
    mu_k = np.load("data/processed/kernel_feature_mu.npy")
    sigma_k = np.load("data/processed/kernel_feature_sigma.npy")
    return W_kernel, mu_k, sigma_k

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

    mu_deep = np.load("data/processed/deep_mlp_feature_mu.npy")
    sigma_deep = np.load("data/processed/deep_mlp_feature_sigma.npy")
    return params, mu_deep, sigma_deep

def compare_models_tabular():
    df_test = pd.read_csv("data/processed/test_data.csv", parse_dates=["timestamp"])
    features = ["mid_price", "spread", "volume_imbalance", "rolling_volatility"]
    df_test = df_test.dropna(subset=features + ["label"])

    # sample 30 rows
    df_small = df_test.sample(n=30, random_state=42).copy().reset_index(drop=True)
    X_vals = df_small[features].values
    y_true = df_small["label"].values

    # logistic
    W_log = np.load("data/processed/model_weights.npy")
    mu_log = np.load("data/processed/feature_mu.npy")
    sigma_log = np.load("data/processed/feature_sigma.npy")
    X_norm_log = (X_vals - mu_log) / sigma_log
    preds_log = predict_logreg(X_norm_log, W_log)

    # kernel
    W_kernel, mu_k, sigma_k = load_kernel_model()
    X_norm_k = (X_vals - mu_k) / sigma_k
    # same params as training
    Z_kernel = random_fourier_features(X_norm_k, D=800, gamma=0.0002, seed=1234)
    preds_kernel = predict_kernel(Z_kernel, W_kernel)

    # deep
    params_deep, mu_d, sigma_d = load_deep_mlp_model()
    X_norm_deep = (X_vals - mu_d) / sigma_d
    preds_deep = predict_deep_mlp(X_norm_deep, params_deep)

    df_display = df_small[["symbol","timestamp"] + features].copy()
    df_display["true_label"] = y_true
    df_display["pred_log"] = preds_log
    df_display["pred_kernel"] = preds_kernel
    df_display["pred_deep"] = preds_deep

    print(df_display.to_string(index=False))

if __name__ == "__main__":
    compare_models_tabular()

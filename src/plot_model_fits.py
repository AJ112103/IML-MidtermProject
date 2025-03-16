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
    mu = np.load("data/processed/deep_mlp_feature_mu.npy")
    sigma = np.load("data/processed/deep_mlp_feature_sigma.npy")
    return params, mu, sigma

def main():
    df = pd.read_csv("data/processed/train_data.csv", parse_dates=["timestamp"])
    features = ["mid_price", "spread", "volume_imbalance", "rolling_volatility"]
    df = df.dropna(subset=features + ["label"])

    # For 2D plotting, let's only use mid_price as X-axis, spread as Y-axis
    X_axis = "mid_price"
    Y_axis = "spread"

    # Limit to a small sample (e.g., 300) so the plot isn't too dense
    df_sample = df.sample(n=300, random_state=42).copy().reset_index(drop=True)

    X_vals = df_sample[X_axis].values
    Y_vals = df_sample[Y_axis].values

    # Build a matrix of all features for the actual prediction
    all_features = df_sample[features].values
    y_true = df_sample["label"].values

    # 1) Logistic model predictions
    W_log, mu_log, sigma_log = load_logistic_model()
    X_norm_log = (all_features - mu_log) / sigma_log
    preds_log = predict_logreg(X_norm_log, W_log)

    # 2) Kernel logistic predictions
    W_kernel, mu_kernel, sigma_kernel = load_kernel_model()
    X_norm_kernel = (all_features - mu_kernel) / sigma_kernel
    Z_kernel = random_fourier_features(X_norm_kernel, D=500, gamma=0.001)
    preds_kernel = predict_kernel(Z_kernel, W_kernel)

    # 3) Deep MLP predictions
    params_deep, mu_deep, sigma_deep = load_deep_mlp_model()
    X_norm_deep = (all_features - mu_deep) / sigma_deep
    preds_deep = predict_deep_mlp(X_norm_deep, params_deep)

    # ----- PLOT 1: Logistic Regression -----
    plt.figure()

    plt.scatter(X_vals, Y_vals, c=preds_log)
    plt.xlabel(X_axis)
    plt.ylabel(Y_axis)
    plt.title("Logistic Regression Predictions")
    plt.show()

    # ----- PLOT 2: Kernel Logistic -----
    plt.figure()
    plt.scatter(X_vals, Y_vals, c=preds_kernel)
    plt.xlabel(X_axis)
    plt.ylabel(Y_axis)
    plt.title("Kernel Logistic Predictions")
    plt.show()

    # ----- PLOT 3: Deep MLP -----
    plt.figure()
    plt.scatter(X_vals, Y_vals, c=preds_deep)
    plt.xlabel(X_axis)
    plt.ylabel(Y_axis)
    plt.title("Deep MLP Predictions")
    plt.show()

if __name__ == "__main__":
    main()

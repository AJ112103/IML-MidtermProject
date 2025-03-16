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

    mu = np.load("data/processed/deep_mlp_feature_mu.npy")
    sigma = np.load("data/processed/deep_mlp_feature_sigma.npy")
    return params, mu, sigma

def load_logistic_model():
    W = np.load("data/processed/model_weights.npy")
    mu = np.load("data/processed/feature_mu.npy")
    sigma = np.load("data/processed/feature_sigma.npy")
    return W, mu, sigma

def ensemble_predict(X):
    # logistic
    W_log, mu_log, sigma_log = load_logistic_model()
    X_log = (X - mu_log) / sigma_log
    preds_log = predict_logreg(X_log, W_log)

    # kernel
    W_kernel, mu_k, sigma_k = load_kernel_model()
    X_k = (X - mu_k) / sigma_k
    Z_k = random_fourier_features(X_k, D=800, gamma=0.0002, seed=1234)
    preds_kernel = predict_kernel(Z_k, W_kernel)

    # deep
    params_deep, mu_d, sigma_d = load_deep_mlp_model()
    X_deep = (X - mu_d) / sigma_d
    preds_deep = predict_deep_mlp(X_deep, params_deep)

    preds_ensemble = []
    for i in range(len(X)):
        votes = [preds_log[i], preds_kernel[i], preds_deep[i]]

        vote_count = { -1:0, 0:0, 1:0 }
        for v in votes:
            vote_count[v] += 1

        best_class = max(vote_count, key=vote_count.get)
        preds_ensemble.append(best_class)

    return np.array(preds_ensemble)

def main():
    df_test = pd.read_csv("data/processed/test_data.csv", parse_dates=["timestamp"])
    features = ["mid_price", "spread", "volume_imbalance", "rolling_volatility"]
    df_test = df_test.dropna(subset=features + ["label"])

    X_test = df_test[features].values
    y_test = df_test["label"].values

    preds = ensemble_predict(X_test)
    accuracy = np.mean(preds == y_test)
    print(f"Ensemble Test Accuracy: {accuracy*100:.2f}%")


if __name__ == "__main__":
    main()

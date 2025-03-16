import numpy as np
import pandas as pd
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


def ensemble_predict(X):
    # Logistic Regression predictions:
    W_log, mu_log, sigma_log = load_logistic_model()
    X_norm_log = (X - mu_log) / sigma_log
    preds_log = predict_logreg(X_norm_log, W_log)
    
    # Kernel Logistic Regression predictions:
    W_kernel, mu_kernel, sigma_kernel = load_kernel_model()
    X_norm_kernel = (X - mu_kernel) / sigma_kernel
    # Apply the random Fourier transform with D=500 and gamma=0.001 as in training
    Z_kernel = random_fourier_features(X_norm_kernel, D=500, gamma=0.001)
    preds_kernel = predict_kernel(Z_kernel, W_kernel)
    
    # Deep MLP predictions:
    params_deep, mu_deep, sigma_deep = load_deep_mlp_model()
    X_norm_deep = (X - mu_deep) / sigma_deep
    preds_deep = predict_deep_mlp(X_norm_deep, params_deep)
    
    # Combine predictions via majority vote:
    combined = []
    for p_log, p_kernel, p_deep in zip(preds_log, preds_kernel, preds_deep):
        votes = [p_log, p_kernel, p_deep]
        # Majority vote; if no majority, fallback to deep MLP prediction
        if votes.count(-1) > 1:
            combined.append(-1)
        elif votes.count(0) > 1:
            combined.append(0)
        elif votes.count(1) > 1:
            combined.append(1)
        else:
            combined.append(p_deep)
    return np.array(combined)

def compute_confusion_matrix(y_true, y_pred, classes=[-1, 0, 1]):
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
    return cm

def compute_classification_metrics(cm, classes=[-1, 0, 1]):
    precision = {}
    recall = {}
    f1_score = {}
    for i, cls in enumerate(classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        precision[cls] = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall[cls] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_score[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls]) if (precision[cls] + recall[cls]) > 0 else 0.0
    return precision, recall, f1_score

def main():
    df_test = pd.read_csv("data/processed/test_data.csv", parse_dates=["timestamp"])
    features = ["mid_price", "spread", "volume_imbalance", "rolling_volatility"]
    df_test = df_test.dropna(subset=features + ["label"])
    X_test = df_test[features].values
    y_test = df_test["label"].values
    
    preds = ensemble_predict(X_test)
    accuracy = np.mean(preds == y_test)
    print(f"Ensemble Test Accuracy: {accuracy * 100:.2f}%")
    
    cm = compute_confusion_matrix(y_test, preds)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=[-1, 0, 1], columns=[-1, 0, 1]))
    
    precision, recall, f1_score = compute_classification_metrics(cm)
    print("\nClassification Metrics:")
    print("Class\tPrecision\tRecall\t\tF1-Score")
    for cls in [-1, 0, 1]:
        print(f"{cls}\t{precision[cls]:.2f}\t\t{recall[cls]:.2f}\t\t{f1_score[cls]:.2f}")

if __name__ == "__main__":
    main()

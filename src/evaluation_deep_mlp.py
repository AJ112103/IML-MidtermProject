import numpy as np
import pandas as pd
from deep_mlp import predict_deep_mlp, normalize_features

def load_deep_mlp_model():
    parameters = {}
    parameters["W1"] = np.load("data/processed/deep_mlp_W1.npy")
    parameters["b1"] = np.load("data/processed/deep_mlp_b1.npy")
    parameters["W2"] = np.load("data/processed/deep_mlp_W2.npy")
    parameters["b2"] = np.load("data/processed/deep_mlp_b2.npy")
    parameters["W3"] = np.load("data/processed/deep_mlp_W3.npy")
    parameters["b3"] = np.load("data/processed/deep_mlp_b3.npy")
    parameters["W4"] = np.load("data/processed/deep_mlp_W4.npy")
    parameters["b4"] = np.load("data/processed/deep_mlp_b4.npy")
    mu = np.load("data/processed/deep_mlp_feature_mu.npy")
    sigma = np.load("data/processed/deep_mlp_feature_sigma.npy")
    return parameters, mu, sigma

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
    parameters, mu, sigma = load_deep_mlp_model()
    X_test_norm = normalize_features(X_test, mu, sigma)
    preds = predict_deep_mlp(X_test_norm, parameters)
    accuracy = np.mean(preds == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    classes = [-1, 0, 1]
    cm = compute_confusion_matrix(y_test, preds, classes=classes)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=classes, columns=classes))
    precision, recall, f1_score = compute_classification_metrics(cm, classes=classes)
    print("\nClassification Metrics:")
    print("Class\tPrecision\tRecall\t\tF1-Score")
    for cls in classes:
        print(f"{cls}\t{precision[cls]:.2f}\t\t{recall[cls]:.2f}\t\t{f1_score[cls]:.2f}")

if __name__ == "__main__":
    main()

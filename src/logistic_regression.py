import numpy as np
import pandas as pd

def softmax(z):
    z_stable = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_loss_and_gradients(X, y_onehot, W, lambda_reg):
    m = X.shape[0]
    z = np.dot(X, W)
    probs = softmax(z)
    loss = -np.sum(y_onehot * np.log(probs + 1e-8)) / m + (lambda_reg / 2) * np.sum(W * W)
    grad = np.dot(X.T, (probs - y_onehot)) / m + lambda_reg * W
    return loss, grad

def train_logistic_regression(X, y, num_epochs=3000, learning_rate=0.001, lambda_reg=0.001):
    m, d = X.shape
    k = 3
    W = np.random.randn(d, k) * 0.01
    y_indices = np.where(y == -1, 0, np.where(y == 0, 1, 2))
    y_onehot = np.eye(k)[y_indices]
    losses = []
    for epoch in range(num_epochs):
        loss, grad = compute_loss_and_gradients(X, y_onehot, W, lambda_reg)
        W -= learning_rate * grad
        losses.append(loss)
        if epoch % 300 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
    return W, losses

def predict(X, W):
    z = np.dot(X, W)
    probs = softmax(z)
    preds_indices = np.argmax(probs, axis=1)
    preds = np.where(preds_indices == 0, -1, np.where(preds_indices == 1, 0, 1))
    return preds

def normalize_features(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return (X - mu) / sigma, mu, sigma

if __name__ == "__main__":
    train_df = pd.read_csv("data/processed/train_data.csv", parse_dates=["timestamp"])
    test_df = pd.read_csv("data/processed/test_data.csv", parse_dates=["timestamp"])
    features = ["mid_price", "spread", "volume_imbalance", "rolling_volatility"]
    train_df = train_df.dropna(subset=features + ["label"])
    test_df = test_df.dropna(subset=features + ["label"])
    X_train = train_df[features].values
    y_train = train_df["label"].values
    X_test = test_df[features].values
    y_test = test_df["label"].values
    X_train, mu, sigma = normalize_features(X_train)
    X_test = (X_test - mu) / sigma
    W, losses = train_logistic_regression(X_train, y_train, num_epochs=3000, learning_rate=0.001, lambda_reg=0.001)
    train_preds = predict(X_train, W)
    train_accuracy = np.mean(train_preds == y_train)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    test_preds = predict(X_test, W)
    test_accuracy = np.mean(test_preds == y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    np.save("data/processed/model_weights.npy", W)
    np.save("data/processed/feature_mu.npy", mu)
    np.save("data/processed/feature_sigma.npy", sigma)
    print("Model training complete and parameters saved.")

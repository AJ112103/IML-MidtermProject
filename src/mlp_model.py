import numpy as np
import pandas as pd

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    Z_stable = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z_stable)
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_x, n_h) * np.sqrt(2.0 / n_x)
    b1 = np.zeros((1, n_h))
    W2 = np.random.randn(n_h, n_y) * np.sqrt(2.0 / n_h)
    b2 = np.zeros((1, n_y))
    return W1, b1, W2, b2

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    cache = (Z1, A1, Z2, A2)
    return A2, cache

def compute_cost(A2, Y, W1, W2, lambda_reg):
    m = Y.shape[0]
    cost = -np.sum(Y * np.log(A2 + 1e-8)) / m + (lambda_reg / 2) * (np.sum(W1 * W1) + np.sum(W2 * W2))
    return cost

def backward_propagation(X, Y, cache, W1, W2, lambda_reg):
    m = X.shape[0]
    Z1, A1, Z2, A2 = cache
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m + lambda_reg * W2
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m + lambda_reg * W1
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, grads, learning_rate, beta, v):
    dW1, db1, dW2, db2 = grads
    v['dW1'] = beta * v['dW1'] + (1 - beta) * dW1
    v['db1'] = beta * v['db1'] + (1 - beta) * db1
    v['dW2'] = beta * v['dW2'] + (1 - beta) * dW2
    v['db2'] = beta * v['db2'] + (1 - beta) * db2
    W1 -= learning_rate * v['dW1']
    b1 -= learning_rate * v['db1']
    W2 -= learning_rate * v['dW2']
    b2 -= learning_rate * v['db2']
    return W1, b1, W2, b2, v

def one_hot_encode(y):
    y_indices = np.where(y == -1, 0, np.where(y == 0, 1, 2))
    return np.eye(3)[y_indices]

def model(X, y, n_h=128, num_epochs=20000, learning_rate=0.001, lambda_reg=0.001, beta=0.9, print_cost=True):
    n_x = X.shape[1]
    n_y = 3
    W1, b1, W2, b2 = initialize_parameters(n_x, n_h, n_y)
    Y = one_hot_encode(y)
    costs = []
    v = {
        'dW1': np.zeros_like(W1),
        'db1': np.zeros_like(b1),
        'dW2': np.zeros_like(W2),
        'db2': np.zeros_like(b2)
    }
    for epoch in range(num_epochs):
        A2, cache = forward_propagation(X, W1, b1, W2, b2)
        cost = compute_cost(A2, Y, W1, W2, lambda_reg)
        grads = backward_propagation(X, Y, cache, W1, W2, lambda_reg)
        W1, b1, W2, b2, v = update_parameters(W1, b1, W2, b2, grads, learning_rate, beta, v)
        if epoch % 2000 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Epoch {epoch}: Cost = {cost:.4f}")
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters, costs

def predict_mlp(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    A2, _ = forward_propagation(X, W1, b1, W2, b2)
    preds_indices = np.argmax(A2, axis=1)
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
    parameters, costs = model(X_train, y_train, n_h=128, num_epochs=20000, learning_rate=0.001, lambda_reg=0.001, beta=0.9, print_cost=True)
    train_preds = predict_mlp(X_train, parameters)
    train_accuracy = np.mean(train_preds == y_train)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    test_preds = predict_mlp(X_test, parameters)
    test_accuracy = np.mean(test_preds == y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    np.save("data/processed/mlp_W1.npy", parameters["W1"])
    np.save("data/processed/mlp_b1.npy", parameters["b1"])
    np.save("data/processed/mlp_W2.npy", parameters["W2"])
    np.save("data/processed/mlp_b2.npy", parameters["b2"])
    np.save("data/processed/mlp_feature_mu.npy", mu)
    np.save("data/processed/mlp_feature_sigma.npy", sigma)
    print("MLP model training complete and parameters saved.")

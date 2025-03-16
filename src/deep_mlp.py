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

def initialize_parameters(n_x, n_h1, n_h2, n_h3, n_y):
    W1 = np.random.randn(n_x, n_h1) * np.sqrt(2.0 / n_x)
    b1 = np.zeros((1, n_h1))
    W2 = np.random.randn(n_h1, n_h2) * np.sqrt(2.0 / n_h1)
    b2 = np.zeros((1, n_h2))
    W3 = np.random.randn(n_h2, n_h3) * np.sqrt(2.0 / n_h2)
    b3 = np.zeros((1, n_h3))
    W4 = np.random.randn(n_h3, n_y) * np.sqrt(2.0 / n_h3)
    b4 = np.zeros((1, n_y))
    return W1, b1, W2, b2, W3, b3, W4, b4

def forward_propagation(X, W1, b1, W2, b2, W3, b3, W4, b4):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = relu(Z3)
    Z4 = np.dot(A3, W4) + b4
    A4 = softmax(Z4)
    cache = (Z1, A1, Z2, A2, Z3, A3, Z4, A4)
    return A4, cache

def compute_cost(A4, Y, W1, W2, W3, W4, lambda_reg):
    m = Y.shape[0]
    cost = -np.sum(Y * np.log(A4 + 1e-8)) / m
    cost += (lambda_reg / 2) * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2) + np.sum(W4**2))
    return cost

def backward_propagation(X, Y, cache, W1, W2, W3, W4, lambda_reg):
    m = X.shape[0]
    Z1, A1, Z2, A2, Z3, A3, Z4, A4 = cache
    dZ4 = A4 - Y
    dW4 = np.dot(A3.T, dZ4) / m + lambda_reg * W4
    db4 = np.sum(dZ4, axis=0, keepdims=True) / m

    dA3 = np.dot(dZ4, W4.T)
    dZ3 = dA3 * relu_derivative(Z3)
    dW3 = np.dot(A2.T, dZ3) / m + lambda_reg * W3
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2) / m + lambda_reg * W2
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m + lambda_reg * W1
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2,
             "dW3": dW3, "db3": db3, "dW4": dW4, "db4": db4}
    return grads

def update_parameters(params, grads, learning_rate, beta, v):
    new_params = {}
    for key in params:
        v[key] = beta * v[key] + (1 - beta) * grads['d' + key]
        new_params[key] = params[key] - learning_rate * v[key]
    return new_params, v

def one_hot_encode(y):
    y_indices = np.where(y == -1, 0, np.where(y == 0, 1, 2))
    return np.eye(3)[y_indices]

def model(X, y, n_h1=256, n_h2=256, n_h3=256, num_epochs=20000, learning_rate=0.001, lambda_reg=0.001, beta=0.9, print_cost=True):
    n_x = X.shape[1]
    n_y = 3
    params = {}
    params["W1"], params["b1"], params["W2"], params["b2"], params["W3"], params["b3"], params["W4"], params["b4"] = initialize_parameters(n_x, n_h1, n_h2, n_h3, n_y)
    Y = one_hot_encode(y)
    costs = []
    v = {key: np.zeros_like(params[key]) for key in params}
    for epoch in range(num_epochs):
        A4, cache = forward_propagation(X, params["W1"], params["b1"], params["W2"], params["b2"],
                                        params["W3"], params["b3"], params["W4"], params["b4"])
        cost = compute_cost(A4, Y, params["W1"], params["W2"], params["W3"], params["W4"], lambda_reg)
        grads = backward_propagation(X, Y, cache, params["W1"], params["W2"], params["W3"], params["W4"], lambda_reg)
        params, v = update_parameters(params, grads, learning_rate, beta, v)
        if epoch % 2000 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Epoch {epoch}: Cost = {cost:.4f}")
    return params, costs

def predict_deep_mlp(X, parameters):
    A4, _ = forward_propagation(X, parameters["W1"], parameters["b1"],
                                parameters["W2"], parameters["b2"],
                                parameters["W3"], parameters["b3"],
                                parameters["W4"], parameters["b4"])
    preds_indices = np.argmax(A4, axis=1)
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
    parameters, costs = model(X_train, y_train, n_h1=256, n_h2=256, n_h3=256,
                               num_epochs=20000, learning_rate=0.001, lambda_reg=0.001,
                               beta=0.9, print_cost=True)
    train_preds = predict_deep_mlp(X_train, parameters)
    train_accuracy = np.mean(train_preds == y_train)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    test_preds = predict_deep_mlp(X_test, parameters)
    test_accuracy = np.mean(test_preds == y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    np.save("data/processed/deep_mlp_W1.npy", parameters["W1"])
    np.save("data/processed/deep_mlp_b1.npy", parameters["b1"])
    np.save("data/processed/deep_mlp_W2.npy", parameters["W2"])
    np.save("data/processed/deep_mlp_b2.npy", parameters["b2"])
    np.save("data/processed/deep_mlp_W3.npy", parameters["W3"])
    np.save("data/processed/deep_mlp_b3.npy", parameters["b3"])
    np.save("data/processed/deep_mlp_W4.npy", parameters["W4"])
    np.save("data/processed/deep_mlp_b4.npy", parameters["b4"])
    np.save("data/processed/deep_mlp_feature_mu.npy", mu)
    np.save("data/processed/deep_mlp_feature_sigma.npy", sigma)
    print("Deep MLP model training complete and parameters saved.")

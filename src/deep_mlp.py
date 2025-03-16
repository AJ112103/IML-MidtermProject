import numpy as np
import pandas as pd

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def apply_dropout(A, p_drop):

    if p_drop <= 0.0:
        return A
    mask = (np.random.rand(*A.shape) > p_drop).astype(float)
    A_dropped = (mask * A) / (1.0 - p_drop)
    return A_dropped

def softmax(Z):
    Z_stable = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z_stable)
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def initialize_parameters(n_x, n_h1, n_h2, n_h3, n_h4, n_y):
    W1 = np.random.randn(n_x, n_h1) * np.sqrt(2.0 / n_x)
    b1 = np.zeros((1, n_h1))
    W2 = np.random.randn(n_h1, n_h2) * np.sqrt(2.0 / n_h1)
    b2 = np.zeros((1, n_h2))
    W3 = np.random.randn(n_h2, n_h3) * np.sqrt(2.0 / n_h2)
    b3 = np.zeros((1, n_h3))
    W4 = np.random.randn(n_h3, n_h4) * np.sqrt(2.0 / n_h3)
    b4 = np.zeros((1, n_h4))
    W5 = np.random.randn(n_h4, n_y) * np.sqrt(2.0 / n_h4)
    b5 = np.zeros((1, n_y))
    return W1, b1, W2, b2, W3, b3, W4, b4, W5, b5

def forward_propagation(X, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, p_drop=0.2):

    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    A1 = apply_dropout(A1, p_drop)

    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    A2 = apply_dropout(A2, p_drop)

    Z3 = np.dot(A2, W3) + b3
    A3 = relu(Z3)
    A3 = apply_dropout(A3, p_drop)

    Z4 = np.dot(A3, W4) + b4
    A4 = relu(Z4)
    A4 = apply_dropout(A4, p_drop)

    Z5 = np.dot(A4, W5) + b5
    A5 = softmax(Z5)
    cache = (Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5)
    return A5, cache

def compute_cost(A5, Y, Ws, lambda_reg):
    m = Y.shape[0]
    cost = -np.sum(Y * np.log(A5 + 1e-8)) / m
    l2_sum = sum([np.sum(W*W) for W in Ws])
    cost += (lambda_reg / 2) * l2_sum
    return cost

def backward_propagation(X, Y, cache, Ws, bs, lambda_reg, p_drop=0.2):
    m = X.shape[0]
    (Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5) = cache
    W1, W2, W3, W4, W5 = Ws

    dZ5 = A5 - Y
    dW5 = np.dot(A4.T, dZ5) / m + lambda_reg * W5
    db5 = np.sum(dZ5, axis=0, keepdims=True) / m

    dA4 = np.dot(dZ5, W5.T)
    dZ4 = dA4 * relu_derivative(Z4)
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

    grads = (dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5)
    return grads

def update_parameters(params, grads, learning_rate, beta, v):
    (dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5) = grads

    for key, dval in zip(["W1","b1","W2","b2","W3","b3","W4","b4","W5","b5"],
                         [dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5]):
        v[key] = beta * v[key] + (1 - beta) * dval
        params[key] -= learning_rate * v[key]

    return params, v

def one_hot_encode(y):
    y_indices = np.where(y == -1, 0, np.where(y == 0, 1, 2))
    return np.eye(3)[y_indices]

def model(X, y,
          n_h1=256, n_h2=256, n_h3=256, n_h4=256,
          num_epochs=20000,
          learning_rate=0.0005,   
          lambda_reg=0.001,
          p_drop=0.2,            
          beta=0.9,
          print_cost=True):
    n_x = X.shape[1]
    n_y = 3

    (W1,b1,W2,b2,W3,b3,W4,b4,W5,b5) = initialize_parameters(n_x,n_h1,n_h2,n_h3,n_h4,n_y)
    params = {"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3,"W4":W4,"b4":b4,"W5":W5,"b5":b5}
    Y = one_hot_encode(y)

    costs = []
    v = {key: np.zeros_like(params[key]) for key in params}

    for epoch in range(num_epochs):
        A5, cache = forward_propagation(X, **params, p_drop=p_drop)
        Ws = [params["W1"], params["W2"], params["W3"], params["W4"], params["W5"]]
        cost = compute_cost(A5, Y, Ws, lambda_reg)
        grads = backward_propagation(X, Y, cache, Ws,
                                     [params["b1"], params["b2"], params["b3"], params["b4"], params["b5"]],
                                     lambda_reg,
                                     p_drop=p_drop)
        params, v = update_parameters(params, grads, learning_rate, beta, v)

        if epoch % 2000 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Epoch {epoch}: Cost = {cost:.4f}")

    return params, costs

def predict_deep_mlp(X, params):
    A5, _ = forward_propagation(X, **params, p_drop=0.0)
    preds_indices = np.argmax(A5, axis=1)
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

    params, costs = model(X_train, y_train,
                          n_h1=256, n_h2=256, n_h3=256, n_h4=256,
                          num_epochs=20000,
                          learning_rate=0.0005,  # lower LR
                          lambda_reg=0.001,
                          p_drop=0.2,
                          beta=0.9,
                          print_cost=True)

    train_preds = predict_deep_mlp(X_train, params)
    train_accuracy = np.mean(train_preds == y_train)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

    test_preds = predict_deep_mlp(X_test, params)
    test_accuracy = np.mean(test_preds == y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    np.save("data/processed/deep_mlp_W1.npy", params["W1"])
    np.save("data/processed/deep_mlp_b1.npy", params["b1"])
    np.save("data/processed/deep_mlp_W2.npy", params["W2"])
    np.save("data/processed/deep_mlp_b2.npy", params["b2"])
    np.save("data/processed/deep_mlp_W3.npy", params["W3"])
    np.save("data/processed/deep_mlp_b3.npy", params["b3"])
    np.save("data/processed/deep_mlp_W4.npy", params["W4"])
    np.save("data/processed/deep_mlp_b4.npy", params["b4"])
    np.save("data/processed/deep_mlp_W5.npy", params["W5"])
    np.save("data/processed/deep_mlp_b5.npy", params["b5"])

    np.save("data/processed/deep_mlp_feature_mu.npy", mu)
    np.save("data/processed/deep_mlp_feature_sigma.npy", sigma)
    print("Deep MLP model training complete and parameters saved.")

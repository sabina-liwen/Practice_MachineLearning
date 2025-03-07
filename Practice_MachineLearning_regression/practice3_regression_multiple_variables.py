# learned from a friend's code 
import numpy as np

np.random.seed(0)
num_samples = 100 

# 1. Generate simulation data w1=5, w2=10, w3=-3, b=20
true_w = np.array([5, 10, -3])
true_b = 20

# feature：size（50~200m²）、rooms（1~5）、age（1~20）
X = np.array([
    np.random.uniform(50, 200, num_samples),     # size
    np.random.randint(1, 6, num_samples),        # rooms
    np.random.randint(1, 20, num_samples)        # age
]).T  # trans (num_samples, 3)

# price of the appartement with noises
y = X.dot(true_w) + true_b + np.random.normal(0, 10, num_samples)  # noise ~ N(0, 10)

# Normalization features (accelerated convergence)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normalized = (X - X_mean) / X_std

# Normalization labels (optional, but usually only Normalization features)
y_mean = np.mean(y)
y_std = np.std(y)
y_normalized = (y - y_mean) / y_std


# 2. Initialization parameters
np.random.seed(42)
w = np.random.randn(3)  # initial weight [w1, w2, w3]
b = 0.0                 # initial bias
learning_rate = 0.01
num_iterations = 1000
loss_history = []

# 3. Gradient descent iteration
for i in range(num_iterations):
    # predicted value：y_hat = X_normalized * w + b
    y_pred = X_normalized.dot(w) + b
    
    # Calculation error (MSE loss)
    error = y_normalized - y_pred
    loss = np.mean(error ** 2)
    loss_history.append(loss)
    
    # Calculate the gradient (chain rule)
    grad_w = -2 * np.mean(error.reshape(-1,1) * X_normalized, axis=0)  # 每个特征对应一个梯度
    grad_b = -2 * np.mean(error)
    
    # Update parameters
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b
    
    # Print training progress
    if i % 100 == 0:
        print(f"Iteration {i:4d}, Loss: {loss:.2f}")

# Inverse Normalization 
w_original_scale = w / X_std * y_std
b_original_scale = (b * y_std + y_mean) - np.dot(X_mean / X_std * y_std, w)

print("True parameters:", true_w, true_b)
print("Learned parameters:", w_original_scale, b_original_scale)



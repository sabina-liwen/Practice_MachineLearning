import numpy as np 

# seeds setting 
np.random.seed(42)

# 1. Simulate house price data
n_samples = 100

# three features
feature1 = size = np.random.randint(80,200,n_samples)
feature2 = age = np.random.randint(5,30,n_samples)
feature3 = rooms = np.random.randint(2,6,n_samples)

# Transpose Matrix
X = np.column_stack([feature1, feature2, feature3])

# Define the true model parameters
w_true = [50000, -150, 2000]
b_true = 200000

# y = X·w + b + noise
y = np.dot(X, w_true) + b_true + np.random.standard_normal(n_samples)

# 2. normalize the data
# Calculate the mean and standard deviation of each feature
X_mean = np.mean(X, axis=0)
X_std = np.std (X, axis=0)
X_normalized =(X - X_mean) / X_std

y_mean = np.mean(y)
y_std = np.std(y)
y_normalized = (y - y_mean) / y_std

# Initialize model parameters
n_feature = X.shape[1]
w = np.zeros(n_feature) 
b = 0
# Set hyperparameters
learning_rate = 0.01
n_iterations = 1000

# save loss history
loss_history = [] 

# 3. Optimizing the model with gradient descent 

for i in range (n_iterations):
    y_pred = np.dot(X_normalized, w) + b
    loss = np.mean((y_pred - y_normalized) ** 2)
    loss_history.append(loss)
    dw = (2/n_samples) * np.dot(X_normalized.T, (y_pred - y_normalized))
    db = (2/n_samples) * np.sum(y_pred - y_normalized)
    w -= learning_rate * dw
    b -= learning_rate * db 
    if i % 100 == 0:
        print (f"iteration:{i}, loss:{loss:.6f}")

# Inverse Normalization 
w_original = w / X_std * y_std
b_original = (b * np.std(y)) + np.mean(y) - np.sum(w_original * X_mean)

# print results
print ("---------finished----------")
print (f"w_size: {w_original[0]:.2f}")
print (f"w_rooms:{w_original[2]:.2f}")
print (f"w_ages:{w_original[1]:.2f}")
print (f"basic price (b):{b_original:.2f}")
print (f"prected price = ({w_original[0]:.2f} * size) + ({w_original[2]:.2f} * rooms) + ({w_original[1]:.2f} * ages) + {b_original:.2f}")
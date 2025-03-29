import numpy as np

np.random.seed(42)
x = size = np.random.randint(50, 201, 50)  # 50~200 m2
y = price_real = np.random.randint(600000, 3000000, 50)
w = np.random.randint(10, 200)  
b = np.random.randint(50000, 200000) 
alpha = 0.0001  
num_iters = 1000 

def compute_cost(x, y, w, b):
    m = len(y)
    f_wb = w * x + b
    error = f_wb - y
    cost = (1 / (2 * m)) * np.sum(error ** 2)
    return cost

def compute_gradient(x, y, w, b):
    m = len(y)
    f_wb = w * x + b
    error = f_wb - y
    dj_dw = (1 / m) * np.sum(error * x)
    dj_db = (1 / m) * np.sum(error)
    return dj_dw, dj_db

def gradient_descent(x, y, w, b, alpha, num_iters):
    cost_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        cost = compute_cost(x, y, w, b)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.2f}, w = {w:.2f}, b = {b:.2f}")

    return w, b, cost_history


w_final, b_final, cost_history = gradient_descent(size, price_real, w, b, alpha, num_iters)

print(f"price_estimated = {w_final:.2f} * size + {b_final:.2f}")

def predict(size, w, b):
    return w * size + b

new_house_size = 150  
predicted_price = predict(new_house_size, w_final, b_final)
print(f"size {new_house_size} m², price_estimated: {predicted_price:.2f} $")
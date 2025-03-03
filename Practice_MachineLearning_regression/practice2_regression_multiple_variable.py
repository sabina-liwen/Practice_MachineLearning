import numpy as np

np.random.seed(42)
x1 = size = np.random.randint(50, 201, 50)  # 50~200 m2
w1 = np.random.randint(10, 200)
x2 = room = np.random.randint(1,5,50)
w2 = np.random.randint(1,50)
y = price_real = np.random.randint(600000, 3000000, 50)
b = np.random.randint(50000, 200000) 
alpha = 0.0001  
num_iters = 1000 

def compute_cost(x1, x2, w1, w2, y, b):
    m = len(y)
    f_wb = w1 * x1 + w2 * x2 + b
    error = f_wb - y
    cost = (1 / (2 * m)) * np.sum(error ** 2)
    return cost

def compute_gradient(x1, x2, w1, w2, y, b):
    m = len(y)
    f_wb = w1 * x1 + w2 * x2 + b
    error = f_wb - y
    dj_dw1 = (1 / m) * np.sum(error * x1)
    dj_dw2 = (1 / m) * np.sum(error * x2)
    dj_db = (1 / m) * np.sum(error)
    return dj_dw1, dj_dw2, dj_db

def gradient_descent(x1, x2, w1, w2, y, b, alpha, num_iters):
    cost_history = []

    for i in range(num_iters):
        dj_dw1, dj_dw2, dj_db = compute_gradient(x1, x2, w1, w2, y, b)
        w1 -= alpha * dj_dw1
        w2 -= alpha * dj_dw2
        b -= alpha * dj_db
        cost = compute_cost(x1, x2, w1, w2, y, b)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.2f}, w1 = {w1:.2f}, w2 = {w2:.2f}, b = {b:.2f}")

    return w1, w2, b, cost_history

w1_final, w2_final, b_final, cost_history = gradient_descent(size, room, w1, w2, price_real, b, alpha, num_iters)

print(f"price_estimated = {w1_final:.2f} * size + {w2_final:.2f} * room + {b_final:.2f}")

def predict(size, room, w1, w2, b):
    return w1 * size + w2 * room + b

new_house_size = 150  
new_house_rooms = 3 
predicted_price = predict(new_house_size, new_house_rooms, w1_final, w2_final, b_final)

print(f"Predicted price for a house of {new_house_size} m² with {new_house_rooms} rooms: {predicted_price:.2f} $")
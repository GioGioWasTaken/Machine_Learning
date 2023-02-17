import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy, math
# Load the data
data = pd.read_csv('train_updated.csv')
print(data[['Id', 'LotFrontage', 'LotArea', 'lot_depth', 'house_age', 'SalePrice']].head(5))

# Fix NaN values in LotFrontage and lot_depth columns
lot_frontage = np.nan_to_num(np.array(data['LotFrontage']), nan=0.1)
lot_depth = np.nan_to_num(np.array(data['lot_depth']), nan=0.1)

# Extract other feature values and standardize them
lot_area = np.array(data['LotArea'])
lot_area = (lot_area - np.mean(lot_area)) / np.std(lot_area)
house_age = np.array(data['house_age'])
house_age = (house_age - np.mean(house_age)) / np.std(house_age)

# Combine feature values into an input matrix
X_train = np.column_stack((lot_frontage, lot_area, lot_depth, house_age))

# Standardize the output values
Y_train = np.array(data['SalePrice'])
Y_train = (Y_train - np.mean(Y_train)) / np.std(Y_train)


def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    for i in range( m ):
        f_wb_i = np.dot( x[i], w ) + b
        cost = cost + (f_wb_i - y[i]) ** 2
    cost = cost / (2 * m)
    return cost

# Define gradient computation function
def compute_gradients(x, y, w, b):
    m, n = x.shape  # (number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
        err = (np.dot( x[i], w ) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * x[i, j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db

# Define gradient descent function
def gradient_descent(x, y, w_init, b_init, alpha, iterations, gradient_function,cost_function):
    w = copy.deepcopy(w_init)
    b = b_init
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    for i in range(1,iterations+1):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if type(i/100)!=float:
            print(f"Iteration {i}# complete.")
        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function( x, y, w, b ) )

            # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil( iterations / 10 ) == 0:
            print( f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   " )

    return w, b, J_history  # return final w,b and J history for graphing

# Initialize weights and bias, and set hyperparameters
w_init = 0.01 * np.random.randn(X_train.shape[1])
b_init = 0
alpha = (1.0e-5)*7.8
iterations = 10000

# Train the model
final_w, final_b,J_hist = gradient_descent(X_train, Y_train, w_init, b_init, alpha, iterations, compute_gradients,compute_cost)
print(f"Final w and b found by gradient descent:\nW: {final_w}\nB: {final_b}")

# Define prediction function
def predict(x, w, b):
    return np.dot(x, w) + b
# observation number to predict from the training data
obsv=0

# Make a prediction for the first training example
prediction = predict(X_train[obsv], final_w, final_b)
print(f"Prediction for first training example: {prediction}\nTraining values: {Y_train[obsv]}")
# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step')
plt.show()


# Iteration 10000: Cost     0.36  , learning rate alpha cannot be made larger without overshooting.
# Final w and b found by gradient descent:
# W: [ 1.82515340e-03  1.30595594e-01  2.27946268e-04 -2.87948989e-01]
# B: -0.06383796734380544
# Prediction for first training example: 0.3600108233179311
# Training values: 0.34727321973650555
# result unsatisfactory, will re-attempt after learning what overfitting is.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy, math
# Load the data
data = pd.read_csv('train_updated.csv')
print(data[['Id','lot_depth', 'house_age', 'SalePrice']].head(5))

# lot depth and house age are features that did not exist before.
# They were made using available data and feature engineering.


# Fix NaN values in LotFrontage and lot_depth columns
lot_frontage = np.nan_to_num(np.array(data['LotFrontage']), nan=0.1)
lot_depth = np.nan_to_num(np.array(data['lot_depth']), nan=0.1)

# Extract other feature values and standardize them
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean( data )
        self.std = np.std( data )

    def transform(self, data):
        return (data - self.mean) / self.std


scaler = StandardScaler()

lot_area = np.array( data['LotArea'] )
house_age = np.array( data['house_age'] )
overall_quality = np.array( data['OverallQual'] )
overall_condition = np.array( data['OverallCond'] )
GarageArea = np.array( data['GarageArea'] )
GarageCars = np.array( data['GarageCars'] )
WoodDeckSF = np.array( data['WoodDeckSF'] )
TotRmsAbvGrd = np.array( data['TotRmsAbvGrd'] )
BedroomAbvGr = np.array( data['BedroomAbvGr'] )
GrLivArea = np.array( data['GrLivArea'] )

scaler.fit( lot_area )
lot_area = scaler.transform( lot_area )

scaler.fit( house_age )
house_age = scaler.transform( house_age )

scaler.fit( overall_quality )
overall_quality = scaler.transform( overall_quality )

scaler.fit( overall_condition )
overall_condition = scaler.transform( overall_condition )

scaler.fit( GarageArea )
GarageArea = scaler.transform( GarageArea )

scaler.fit( GarageCars )
GarageCars = scaler.transform( GarageCars )

scaler.fit( WoodDeckSF )
WoodDeckSF = scaler.transform( WoodDeckSF )

scaler.fit( TotRmsAbvGrd )
TotRmsAbvGrd = scaler.transform( TotRmsAbvGrd )

scaler.fit( BedroomAbvGr )
BedroomAbvGr = scaler.transform( BedroomAbvGr )

scaler.fit( GrLivArea )
GrLivArea = scaler.transform( GrLivArea )

# Combine feature values into an input matrix
X_train = np.column_stack((lot_frontage, lot_area, lot_depth, house_age,overall_quality,overall_condition,GarageArea, GarageCars, WoodDeckSF, TotRmsAbvGrd, BedroomAbvGr, GrLivArea))
print(X_train.shape)
# Standardize the output values
Y_train = np.array(data['SalePrice'])
Y_train = (Y_train - np.mean(Y_train)) / np.std(Y_train)


def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    f_wb = np.dot(X, w) + b
    cost =np.sum((f_wb - y) ** 2)/ (2 * m)
    return cost

# Define gradient computation function
def compute_gradients(X, y, w, b):
    m, n = X.shape  # (number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.
    err = (np.dot(X, w) + b) - y
    dj_dw = np.dot(X.T, err) / m
    # this is quite complicated, so let's try to understand it with this comment
    # X was originally an array with dimensions (1460,12) 1460 observations, 12 features
    # we now changed it to its transposition (12,1460) which is an array that has 12 observations, 1460 features
    # this works well, because each observation of X.T, is an array of all the observation at the J position
    # for example, the first observation of X.T would be an array of all the 1st features across all observations of X.
    # so what we are essentially making now, is an array dj_dw that has (n,) dimensions
    # where each item, is the dot product of the error, and all the features at the jth position.
    dj_db = np.sum(err) /m
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
print(f"Prediction for #{obsv} training example: {prediction}\nTraining values: {Y_train[obsv]}")

def predict_all(x,y,w,b):
    all_pred=[]
    all_pred.append(np.dot(x,w)+b)
    return all_pred
all_pred=predict_all(X_train,Y_train,final_w,final_b)
print(f"Training data: {list(Y_train)[0:100]}\nPredictions: {all_pred[0:100]}")


# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step')
plt.show()
# Third plot: Scatter plot of all data with final model
# Deduction from current results: the more features, the better this model works.
# Perfect for practice.
#Iteration 10000: Cost     0.21
# Final w and b found by gradient descent:
# W: [ 1.11862456e-03  1.24833782e-01  2.69276287e-04 -2.04355404e-01
#   3.74552936e-01  1.99124144e-03]
# B: -0.05175613924547292
# Prediction for #0 training example: 0.48786194512065045
# Training values: 0.34727321973650555
# Training data: [ 0.34727322  0.00728832  0.53615372 ...  1.07761115 -0.48852299
#  -0.42084081]
# Predictions: [ 0.48786195  0.06803958  0.52225555 ...  0.08579114 -0.38823288
#  -0.2789255 ]

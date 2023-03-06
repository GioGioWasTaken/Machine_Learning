import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import copy, math
scaler = StandardScaler()

data=pd.read_csv('train.csv')

# extract features
cyro_sleep=np.array(data['CryoSleep'])
cyro_sleep=np.where(cyro_sleep, 1, 0)
age=np.array(data['Age'])
VIP=np.array(data['VIP'])
VIP=np.where(VIP,1,0)
room_service=np.array(data['RoomService'])
food_court=np.array(data['FoodCourt'])
shopping_mall=np.array(data['ShoppingMall'])
spa=np.array(data['Spa'])
VRDeck=np.array(data['VRDeck'])

Y_train=np.where(np.array(data['Transported']),1 , 0)
X_train=np.column_stack((cyro_sleep,age,VIP,room_service,food_court,shopping_mall, spa, VRDeck))
print(data[['CryoSleep', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])
def sigmoid(z):
    prob = 1/ (1+ np.exp(-z))
    return prob

def cost_function(x, y, w, b):
    cost = 0
    m = x.shape[0]
    z = np.dot(x, w) + b
    g_z = sigmoid(z)
    cost = -y * np.log(g_z + 1e-9) - (1 - y) * np.log(1 - g_z + 1e-9) # calculating the cost + adding a small constant to prevent NAN.
    cost = cost / m
    return cost
X_train_scaled = scaler.fit_transform(X_train)
w_init=np.zeros_like(X_train_scaled[0])
b_init=0.
print(f"Starting w and b: {w_init,b_init}\nType of w: {type(w_init)}")
def calc_gradient(x,y,w,b):
    m=x.shape[0]
    z=np.dot(x,w)+b
    print(f"z: {z}")
    g_z=sigmoid(z)
    print(f"g_z: {g_z}")
    err= y - g_z
    print(f"err: {err}")
    dj_dw=np.dot(x.T,err) /m
    dj_db=np.sum(err)/m
    return dj_dw, dj_db
def gradient_descent(x,y,w_init,b_init,alpha,iterations):
    w = copy.deepcopy(w_init)
    b = b_init
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    for i in range(1, iterations + 1):
        dj_dw, dj_db = calc_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        #print(f"djdw:{dj_dw}\ndjdb:{dj_db}")
        # Save cost J at each iteration
        if i < iterations:  # prevent resource exhaustion
            J_history.append(cost_function(x, y, w, b))
            # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(iterations / 10) == 0:
            print(f"Iteration:{i} Cost: {J_history}   ")
    return dj_dw, dj_db,J_history
alpha= 0.1
iterations=3
final_w,final_b,J_hist=gradient_descent(X_train_scaled,Y_train,w_init,b_init,alpha,iterations)
print(final_w,final_b)
#fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
#ax1.plot(J_hist)
#ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
#ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
#ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost')
#ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step')
#plt.show()
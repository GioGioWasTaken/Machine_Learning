import numpy as np
# X_train[size,bedrooms,floors,house_age]
X_train=np.array([2104,5,1,45])
y_train=np.array([460])
w_init=np.array([0.39 ,18.75,-53.36,-26.42])
b_init=785.1811367994083
def prediction(X,w,b):
    n=X.shape[0]
    for i in range(n):
        p=np.dot(X,w)+b
    return p

print(prediction(X_train,w_init,b_init))
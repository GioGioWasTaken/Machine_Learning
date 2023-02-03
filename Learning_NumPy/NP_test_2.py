import numpy as np

a = np.array([1, 2, 3]) # a is the frontage
b = np.array([4, 5, 6]) # b is the depth
# let's now make a third feature c, that will be the area, using feature engineering.
c = a*b
X_train=np.zeros((3,3))
X_train[0,0:3]=a
X_train[1,0:3]=b
X_train[2,0:3]=c
print(X_train) #X_train should now have all 3 features arranged in a 2D list
W=np.random.rand(3,3)
print(W)
b=50
F_wb=np.dot(X_train,W)+b
print(f"The resulting prediction is: \n{F_wb}")
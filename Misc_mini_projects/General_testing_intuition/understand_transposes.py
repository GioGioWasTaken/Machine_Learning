import numpy as np

X = np.array([[1, 2, 3], [4, 5, 6], [4, 5, 6]])

theta = np.array([1, 2, 3])

gradient= np.dot(X.T,theta)
print(gradient,gradient.shape)

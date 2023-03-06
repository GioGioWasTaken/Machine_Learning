import numpy as np
a= np.array([[1,2,3],[4,5,6]])
print(a,a.shape)
print(a.reshape(a.shape[1],-1))
print(a.T)
print(a)
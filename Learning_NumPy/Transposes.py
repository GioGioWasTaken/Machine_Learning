import numpy as np
a= np.array([[1,2,3,4],
             [5,4,6,7],
             [8,9,10,11]])
print(a.shape)
prediction=np.ones((12,))
resultA=np.dot(a.T,prediction)
resultB=np.dot(a,prediction)
print(resultA)
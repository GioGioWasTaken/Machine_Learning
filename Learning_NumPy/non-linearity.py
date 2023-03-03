import numpy as np
a=np.array([[1,2], [2,3], [4,5]])
a[:,1]=a[:,1]**2
print(a)
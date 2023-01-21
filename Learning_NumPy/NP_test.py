import numpy as np
b=np.array([[9.0,8.0,7.0],[6.0,5.0,4.0]])
print(b.ndim) #amounts of dimenions in the array
print(b.dtype)
print(b.itemsize) #amount of bytes
#How to get a specific element? array[row,column] a row would be an observation
#while a column is a feature.
print(b[-1,-1]) # prints the last feature last observation
print(b[0,1:6:2]) #numpy follows the same rules as index slicing.
b[0,0]=5
print(b[0,0]) #numpy arrays are mutable.

b[:,0]=[1,2]  #this converts every feature in the 0th column to 1,2.
print(b)
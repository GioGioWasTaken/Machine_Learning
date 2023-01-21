import numpy as np
a=np.zeros((150,4)) # 150 observations, 4 features. initialized as all zeros.
# this can also be done for any other number. Example:
b=np.full((2,4,4),50,dtype='int32')
#syntax: np.full((shape),wanted_number,dtype)
# this 3 dimensional list is ordered like this:
# 2 big lists->4 observations->4 features
print(b)
#we can also make arrays out of existing arrays.
example=np.zeros((2,2))
print(example)#example is the element itself
print(example.shape) #while example.shape is a way to describe the element's different dimensions

child_array=np.full(example.shape,20) #note that I made the term 'child array' up.
print(child_array)
e=np.full((4,2,2),25)
print(e)
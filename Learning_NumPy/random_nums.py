import numpy as np
arr=np.random.uniform(-10,10,(5,3)) # random numbers between -10->10, assigned a shape.
#making a specific array
arr2=np.ones((5,5))
arr2[1:-1,1:-1]=0
arr2[2,2]=9
print(arr2)
# NEVER COPY AN ARRAY (without .copy() )
a=np.array([1,2,3])
b=a
b[0]=100
print(f"a={a}\nb={b}")
#changes both outputs! to prevent that, add .copy() to b.
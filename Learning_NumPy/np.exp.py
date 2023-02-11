import numpy as np
import matplotlib.pyplot as plt
a= np.array([1,2,3,4,5])
print(np.exp(a))
# np.exp will take an input, and output e to the power of that input.
def sigmoid_func(z):
    g=1/(1+np.exp(-z))
    return g
    # 1 divided by (1+e^-z)


z_tmp=np.arange(1,5)
y=sigmoid_func(z_tmp)
print(z_tmp,y)

# Plot the arrays
plt.plot(z_tmp, y, '-o')

# Add a title and axis labels+show it
plt.title("Sigmoid function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()
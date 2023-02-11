import numpy as np

x=np.arange(1,26)
x=x.reshape(5,5)
print(x) # a fast way to make arrays that go up by 1 for each term.
prec_test=np.array([0.5,0.05,0.005,5,50,500,5e+6])
np.set_printoptions(precision=8)
print(prec_test) # I don't understand wth precision is supposed to do.
# I thought it might refer to the amount of digits shown but ;=;
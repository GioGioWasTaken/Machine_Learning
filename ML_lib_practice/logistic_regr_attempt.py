import numpy as np
def sigmoid(z,f_wb):
    g=1/1+np.exp(-f_wb)
    if g>0.5:
        return 1
    else:
        return 0
x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])
w_init=np.zeros(len(x_train))
b_init=0
f_wb=np.dot(x_train,w_init) + b_init
# ok i'm kind of stuck on how much the model should be adjusted (because it rounds up)
# still, better than not trying.

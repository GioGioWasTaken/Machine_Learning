import numpy as np
import matplotlib as plt
x_train = np.array([1.0, 2.0])       # features
y_train = np.array([300.0, 500.0])   # target value
def compute_gradient(x,y,w,b):
    m=x.shape[0]
    dj_dw=0
    dj_db=0
    for i in range(m):
        f_wb=x[i]*w + b
        dj_dw_i=(f_wb-y[i])*x[i]
        dj_db_i=(f_wb-y[i])
        dj_dw+=dj_dw_i
        dj_db+=dj_db_i
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw,dj_db
def gradient_descent(x,y,w_init,b_init,gradient_function,alpha,num_iters):
    w=w_init
    b=b_init
    for i in range(num_iters):
        dj_dw,dj_db=gradient_function(x,y,w,b)
        w=w-dj_dw*alpha
        b=b-dj_db*alpha
        print(f"I'm w: {w}\nI'm b: {b}")
    return w,b
# gradient descent settings
alpha=1.0e-2
iterations=10000
w_init=0
b_init=0
w_final,b_final=gradient_descent(x_train,y_train,w_init,b_init,compute_gradient,alpha,iterations)
print(f"Final W: {w_final}\nFinal b: {b_final}")
x_test=2
prediction=w_final*x_test +b_final
print(prediction)
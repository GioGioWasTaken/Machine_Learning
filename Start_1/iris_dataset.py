from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pylab as plt
#print(load_iris(return_X_y=True))
X,y=load_iris(return_X_y=True)
model=KNeighborsRegressor().fit(X,y)
pipe=Pipeline([
    ("scale",StandardScaler()),("model",KNeighborsRegressor())
])

pipe.fit(X, y)
pred=pipe.predict(X)
plt.scatter(pred,y)
plt.show()
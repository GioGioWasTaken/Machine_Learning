import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss
from sklearn.metrics import log_loss

# Load data
data = pd.read_csv('train.csv')

# Extract features
cyro_sleep = np.array(data['CryoSleep'])
cyro_sleep = np.where(cyro_sleep, 1, 0)
age = np.array(data['Age'])
VIP = np.array(data['VIP'])
VIP = np.where(VIP, 1, 0)
room_service = np.array(data['RoomService'])
food_court = np.array(data['FoodCourt'])
shopping_mall = np.array(data['ShoppingMall'])
spa = np.array(data['Spa'])
VRDeck = np.array(data['VRDeck'])

Y_train = np.where(np.array(data['Transported']), 1, 0)
X_train = np.column_stack((cyro_sleep, age, VIP, room_service, food_court, shopping_mall, spa, VRDeck))

# Define pipeline to scale data and handle missing values
preprocessing_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='median'))
])

# Fit pipeline on training data and transform X_train
X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train)

# Define logistic regression model and fit on preprocessed data
model = LogisticRegression(max_iter=100)
model.fit(X_train_preprocessed, Y_train)

# Print model coefficients and intercept
print(f"Model weights: {model.coef_}")
print(f"Model bias: {model.intercept_}")

predictions=model.predict(X_train_preprocessed)
print(f"Predictions:{list(predictions)}\nTrueValues:{list(Y_train)}")

# compute the hamming loss and the log loss
hamming_distance = hamming_loss(Y_train, predictions)

Y_train_proba = model.predict_proba(X_train_preprocessed)
log_loss_train = log_loss(Y_train, Y_train_proba)

print(f"Hamming distance: {hamming_distance}")
print(f"Log loss: {log_loss_train}")

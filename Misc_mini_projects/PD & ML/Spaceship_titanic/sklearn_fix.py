import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss
from sklearn.metrics import log_loss
import category_encoders as ce


# Load data
training_data = pd.read_csv('train.csv')
testing_data=pd.read_csv('test_spaceship.csv')

# Extract features
def take_features(data_source):
    cyro_sleep = np.array(data_source['CryoSleep'])
    cyro_sleep = np.where(cyro_sleep, 1, 0)
    age = np.array(data_source['Age'])
    VIP = np.array(data_source['VIP'])
    VIP = np.where(VIP, 1, 0)
    room_service = np.array(data_source['RoomService'])
    food_court = np.array(data_source['FoodCourt'])
    shopping_mall = np.array(data_source['ShoppingMall'])
    spa = np.array(data_source['Spa'])
    VRDeck = np.array(data_source['VRDeck'])
    # we will now make features a bit more complicated, as they are not a binary condition.
    HomePlanet=np.array(data_source['HomePlanet'])
    Destination=np.array(data_source['Destination'])
    # Define the conditions for HomePlanet
    is_europe = HomePlanet == 'Europa'
    is_earth = HomePlanet == 'Earth'
    is_mars = HomePlanet == 'Mars'
    is_blank = HomePlanet == ''
    # Define the conditions for Destination
    is_trappist = Destination == 'TRAPPIST-1e'
    is_cancri = Destination == '55 Cancri e'
    is_PS0 = Destination == 'PSO J318.5-22'
    is_blank2 = Destination == ''
    # Use nested np.where statements to apply the conditions
    HomePlanet = np.where(is_europe, 1, np.where(is_earth, 2, np.where(is_mars, 3, 0)))
    Destination = np.where(is_trappist, 1, np.where(is_cancri, 2, np.where(is_PS0, 3, 0)))

    # Extract cabin information

    # Extract deck information
    data_source['Deck'] = data_source['Cabin'].str.slice(stop=1)
    deck_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    data_source['Deck'] = data_source['Deck'].map(deck_mapping)
    data_source['Deck'] = data_source['Deck'].fillna(0).astype(int)
    Deck=np.array(data_source['Deck'])
    # Extract number information
    data_source['CabinNumber'] = data_source['Cabin'].str.extract('(\d+)')
    data_source['CabinNumber'] = data_source['CabinNumber'].fillna(0).astype(int)
    CabinNumber=np.array(data_source['CabinNumber'])
    # Extract side information
    Side = np.where(data_source['Cabin'].str.slice(-1) == 'P', 1, 0)
    return cyro_sleep, age, VIP, room_service, food_court,  shopping_mall, spa, VRDeck, HomePlanet, Destination, Deck, CabinNumber, Side


cyro_sleep, age, VIP, room_service, food_court,  shopping_mall, spa, VRDeck, HomePlanet, Destination, Deck, CabinNumber, Side = take_features(training_data)


Y_train = np.where(np.array(training_data['Transported']), 1, 0)
X_train = np.column_stack((cyro_sleep, age, VIP, room_service, food_court,  shopping_mall, spa, VRDeck, HomePlanet, Destination, Deck, CabinNumber, Side))
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

# Testing data, for submission:
cyro_sleep_test, age_test, VIP_test, room_service_test, food_court_test,  shopping_mall_test, spa_test, VRDeck_test, HomePlanet_test, Destination_test, Deck_test, CabinNumber_test, Side_test=take_features(testing_data)

X_test=np.column_stack((cyro_sleep_test, age_test, VIP_test, room_service_test, food_court_test,  shopping_mall_test, spa_test, VRDeck_test, HomePlanet_test, Destination_test, Deck_test, CabinNumber_test, Side_test))
Passanger_id_test=np.array(testing_data['PassengerId'])
X_test_preprocessed = preprocessing_pipeline.transform(X_test)
# making a prediction for the testing data:
test_predictions = model.predict(X_test_preprocessed)
test_predictions = np.where(test_predictions == 1, True, False)
print(f"Test Predictions: {list(test_predictions)}")
submission = pd.DataFrame({'PassengerId': Passanger_id_test, 'Transported': test_predictions})
submission.to_csv('Spaceship_Titanic_submissions.csv', index=False)
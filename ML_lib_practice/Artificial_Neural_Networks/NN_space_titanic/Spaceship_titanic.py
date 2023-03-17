import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# Load data
training_data = pd.read_csv('train.csv')
testing_data=pd.read_csv( 'test_spaceship.csv' )

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
    # Define the conditions for Destination
    is_trappist = Destination == 'TRAPPIST-1e'
    is_cancri = Destination == '55 Cancri e'
    is_PS0 = Destination == 'PSO J318.5-22'
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
    group_size=np.array(data_source['PassengerId'].str.slice(-1))
    is_child=np.where(data_source['Age']>=15, 1, 0)
    return cyro_sleep, age, VIP, room_service, food_court,  shopping_mall, spa, VRDeck, HomePlanet, Destination, Deck, CabinNumber, Side, group_size, is_child


cyro_sleep, age, VIP, room_service, food_court,  shopping_mall, spa, VRDeck, HomePlanet, Destination, Deck, CabinNumber, Side, group_size, is_child = take_features(training_data)


Y_train = np.where(np.array(training_data['Transported']), 1, 0)
X_train = np.column_stack((cyro_sleep, age, VIP, room_service, food_court,  shopping_mall, spa, VRDeck, HomePlanet, Destination, Deck, CabinNumber, Side , group_size, is_child))

preprocessing_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='median'))
])

# Fit pipeline on training data and transform X_train
X_train = preprocessing_pipeline.fit_transform(X_train)

#Save the ndarray as a different variable
Y_train_numpy=Y_train
# convert the arrays to tensorflow objects
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)


model=tf.keras.models.Sequential([
    tf.keras.layers.Dense( 80, activation="relu" ),
    tf.keras.layers.Dense( 40, activation="relu" ),
    tf.keras.layers.Dense( 20, activation="relu" ),
    tf.keras.layers.Dense( 10, activation="relu" ),
    tf.keras.layers.Dense( 5, activation="relu" ),
    tf.keras.layers.Dense( 1, activation="sigmoid" )
])
model.compile(loss=tf.keras.losses.binary_crossentropy,optimizer=tf.keras.optimizers.Adam(0.01))

history = model.fit(X_train,Y_train,epochs=200)

predictions=model.predict(X_train)
predictions = np.array(predictions).flatten()
predictions=np.where(predictions>=0.5, 1 , 0)

hamming_distance = tf.math.reduce_mean(tf.cast(tf.math.not_equal(Y_train, predictions), dtype=tf.float32))
print(f"Hamming distance: {hamming_distance}")
print(f"Predictions:{predictions.tolist()}\nTrueValues:{list(Y_train_numpy)}")

# plot the loss over epochs
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

cyro_sleep_test, age_test, VIP_test, room_service_test, food_court_test,  shopping_mall_test, spa_test, VRDeck_test, HomePlanet_test, Destination_test, Deck_test, CabinNumber_test, Side_test, group_size_test, is_child_test=take_features(testing_data)

X_test=np.column_stack((cyro_sleep_test, age_test, VIP_test, room_service_test, food_court_test,  shopping_mall_test, spa_test, VRDeck_test, HomePlanet_test, Destination_test, Deck_test, CabinNumber_test, Side_test, group_size_test, is_child_test))
Passanger_id_test=np.array(testing_data['PassengerId'])
X_test = preprocessing_pipeline.transform(X_test)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

test_predictions=model.predict(X_test)
print(test_predictions)
test_predictions = np.array(test_predictions).flatten()
print(test_predictions)
test_predictions = np.where(test_predictions>=0.5, True, False)

submission = pd.DataFrame({'PassengerId': Passanger_id_test, 'Transported': test_predictions})
submission.to_csv('Spaceship_Titanic_submissions_NeuralNetworks.csv', index=False)
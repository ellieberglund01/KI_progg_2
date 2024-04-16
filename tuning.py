import tensorflow as tf
from tensorflow import keras
import kerastuner as kt
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from Config import *


#Do not need this 

# Define your neural network as a HyperModel
class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(SIZE**2+SIZE **2+1,)))
        # Add hidden layers
        for i in range(hp.Int('num_layers', 1, 3)):  # Search for the number of hidden layers
            model.add(keras.layers.Dense(units=hp.Choice('hidden_units_' + str(i), [32,64, 128, 256, 512]), activation='relu')) #Search for number of neurons
        model.add(keras.layers.Dense(SIZE**2, activation='softmax'))
        
        # Compile the model
        model.compile(optimizer=keras.optimizers.Adam(hp.Float('learning_rate',1e-4, 1e-2, sampling='log')), 
                      loss='categorical_crossentropy',
                      metrics=['accuracy']) #Search for learning_rate
        return model

# Load your data
RBUF = pickle.load(open('RBUF_game_cases3.pkl', "rb"))
board_state = []
distribution = []
for element in RBUF:
    board_state.append(element[0])
    distribution.append(np.array(element[1])) 
board_state = np.array(board_state)
distribution = np.array(distribution) 

X_train, X_val, y_train, y_val = train_test_split(board_state, distribution, test_size=0.3, random_state=42)

# Create an instance of your HyperModel
hypermodel = MyHyperModel()

# Choose a tuner and specify the search space
tuner = kt.Hyperband(hypermodel,
                     objective='val_accuracy',
                     max_epochs=20,
                     factor=3,
                     #directory='my_dir',
                     #project_name='tuning')
                    )

# Run the hyperparameter search
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Retrain the model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

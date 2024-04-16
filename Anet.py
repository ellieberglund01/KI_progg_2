import tensorflow as tf
from keras import layers
from keras.layers import Dense, Input
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random

class NeuralNetwork:
    def __init__(self, activation_function, hidden_layers, learning_rate, optimizer, epochs, board_size): 
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.hidden_layers = hidden_layers
        self.optimizer = optimizer
        self.epochs = epochs
        self.board_size = board_size
        self.epsilon = 1
        self.model = self.build_model()
        self.anet_version = ''

    #Builds neural network
    def build_model(self):
        model = tf.keras.models.Sequential() #groups a linear stack of layers into a tf.keras.Model.
        
        #Change PID to 0 and 1 
        model.add(tf.keras.layers.Input(shape=(self.board_size**2+self.board_size**2+1,))) #Input layer: 2*Board state^2 + PID
        for neurons in self.hidden_layers:
            model.add(tf.keras.layers.Dense(neurons, activation=self.activation_function,)) #Hidden layers
        model.add(tf.keras.layers.Dense(self.board_size**2, activation=tf.keras.activations.softmax,)) #Output layer

        #We specify the training configuration (optimizer, loss, metrics):
        model.compile(optimizer = self.optimizer, loss = 'mse', metrics = ['accuracy']) #Gjøre om til globale konstander i config
        model.summary()
        return model
    
    def fit(self, minibatch: list): #Minibatch from RBUF (s,D). Fit funciton adjusts model parameters and minimize loss.
        board_state = []
        distribution = []
        for element in minibatch:
            board_state.append(element[0]) #board state + PID
            distribution.append(np.array(element[1])) #distribution of possible actions

        board_state = np.array(board_state) #flatten hex board. x_train + PID
        distribution = np.array(distribution) #distribution list. y_train
        self.model.fit(board_state, distribution, batch_size=32, epochs=self.epochs) # her er den innebygde funksjonen fit. Tar inn x_train, y_train, epochs


    #Argmax prediction
    def predict(self, valid_and_invalid_actions, game): 
        board_state = np.array(game.board).flatten()
        board_state = np.insert(board_state, 0, game.player_turn)
        board_state = tf.convert_to_tensor([board_state])

        output = self.model.predict(board_state).flatten()

        for i in range(len(valid_and_invalid_actions)): 
            if valid_and_invalid_actions[i] == 0:
                output[i] = 0
        output = self.custom_soft_max(output)
        max_index = np.argmax(output)
        return valid_and_invalid_actions[max_index] 
    
    #Functions to select action based on some randomness (only in the beginning of the each gam?) 
    def select_best_move_random(self, valid_and_invalid_actions, game):
        if random.random() < 0.2:
            return self.predict(valid_and_invalid_actions, game)
        return self.choose_probabilistic(valid_and_invalid_actions, game)

    #Bruker denne helt i starten av spillet (3 første move) i TOPP for å så argmax
    def choose_probabilistic(self, valid_and_invalid_actions, game):
        board_state = np.array(game.board).flatten()
        board_state = np.insert(board_state, 0, game.player_turn) 
        board_state = tf.convert_to_tensor([board_state])

        output = self.model.predict(board_state).flatten()
        for i in range(len(valid_and_invalid_actions)): #if action is invalid, set output to 0
            if valid_and_invalid_actions[i] == 0:
                output[i] = 0
        output = self.custom_soft_max(output)
        out = [0]
        for prob in output:
            out.append(out[-1] + prob)
        r = random.random()
        for i in range(1, len(out)):
            if r < out[i]:
                return valid_and_invalid_actions[i-1]


    #Custom softmax function so 0 values are still 0
    def custom_soft_max(self, arr):
        non_zero_indices = np.where(arr != 0)[0]
        non_zero_values = arr[non_zero_indices]
        exp_values = np.exp(non_zero_values - np.max(non_zero_values))
        softmax_values = exp_values / np.sum(exp_values)
        result = np.zeros_like(arr)
        result[non_zero_indices] = softmax_values
        return result

    def save_weights(self, filename):
        self.model.save_weights(filename)
        self.anet_version = filename

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def restart_epsilon(self):
        self.epsilon = 1
        
    def get_epsilon(self):
        return self.epsilon
    
    def shrink_epsilon(self):
        self.epsilon = self.epsilon * 0.99



def train_all_data():
    RBUF = pickle.load(open('RBUF_game_cases3.pkl', "rb"))
    anet = NeuralNetwork(activation_function='relu', hidden_layers=[128,128,128], learning_rate=0.001, optimizer='adam', epochs=150, board_size=7) 
    board_state = []
    distribution = []
    for element in RBUF:
        board_state.append(element[0]) #board state + PID
        distribution.append(np.array(element[1])) #distribution of possible actions
    board_state = np.array(board_state) #flatten hex board. x_train + PID
    distribution = np.array(distribution) #distribution list. y_train
    X_train, X_val, y_train, y_val = train_test_split(board_state, distribution, test_size=0.3, random_state=42)
    history = anet.model.fit(X_train, y_train, epochs=150, batch_size=128) 
    test_loss, test_acc = anet.model.evaluate(X_val, y_val)
    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)

    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()





import tensorflow as tf
from keras import layers
from keras.layers import Dense, Input
import numpy as np
import pickle

print("TensorFlow version:", tf.__version__)

class NeuralNetwork:
    def __init__(self, activation_function, hidden_layers, learning_rate, optimizer, epochs, board_size): 
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.hidden_layers = hidden_layers
        self.optmizer = optimizer
        self.epochs = epochs
        self.board_size = board_size
        self.epsilon = 1
        self.model = self.build_model()

    #Builds neural network
    def build_model(self):
        model = tf.keras.models.Sequential() #groups a linear stack of layers into a tf.keras.Model.
        #Change PID to 0 and 1 
        model.add(tf.keras.layers.Input(shape=(self.board_size**2+self.board_size**2+1,))) #Input layer. Board state + PID. Mulig denne kun tar inn integers og ikke tuples?
        for neurons in self.hidden_layers:
            model.add(tf.keras.layers.Dense(neurons, activation=self.activation_function,)) #Hidden layers
        model.add(tf.keras.layers.Dense(self.board_size**2, activation=tf.keras.activations.softmax,)) #Output layer. Må vi ha softmax?

        #We specify the training configuration (optimizer, loss, metrics):
        model.compile(optimizer = 'adam', loss = ['mse'], metrics = ['accuracy'])
        model.summary()
        return model
    
    def fit(self, minibatch: list): #Minibatch from RBUF (s,D). Fit funciton adjusts model parameters and minimize loss. s inneholder PID
        board_state = []
        distribution = []
        for element in minibatch:
            board_state.append(element[0]) #board state + PID
            distribution.append(np.array(element[1])) #distribution of possible actions

        board_state = np.array(board_state) #flatten hex board. x_train + PID
        distribution = np.array(distribution) #distribution list. y_train
        self.model.fit(board_state, distribution, batch_size=32, epochs=self.epochs) # her er den innebygde funksjonen fit. Tar inn x_train, y_train, epochs


    def predict(self, valid_and_invalid_actions, game): #Predicts x_train. Return best move or distribution on correct format
        board_state = np.array(game.board).flatten()
        board_state = np.insert(board_state, 0, game.player_turn) #hvordan vet man at den sjekker PID i tening og predict?
        board_state = tf.convert_to_tensor([board_state])
        print("board_state", board_state)
        output = self.model.predict(board_state).flatten()
        print("valid_and_invalid_actions", valid_and_invalid_actions)
        for i in range(len(valid_and_invalid_actions)): #if action is invalid, set output to 0
            if valid_and_invalid_actions[i] == 0:
                output[i] = 0
        print(output)
        output = self.custom_soft_max(output)
        print("Normalized output", output)
        max_index = np.argmax(output)
        return valid_and_invalid_actions[max_index] #returns best move?
    
    #Custom softmax function so 0 values are still 0
    def custom_soft_max(self, arr):
        # Find non-zero indices and values
        non_zero_indices = np.where(arr != 0)[0]
        non_zero_values = arr[non_zero_indices]
        # Compute softmax only on non-zero values
        exp_values = np.exp(non_zero_values - np.max(non_zero_values))
        softmax_values = exp_values / np.sum(exp_values)
        # Initialize result array with zeros
        result = np.zeros_like(arr)
        # Assign softmax values to non-zero indices
        result[non_zero_indices] = softmax_values
        return result

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)

    def restart_epsilon(self):
        self.epsilon = 1 #Burde kunne restarte til en annen epsilon enn 1

    def get_epsilon(self):
        return self.epsilon
    
    def shrink_epsilon(self):
        self.epsilon = self.epsilon * 0.99


def train_all_data():
    RBUF = fetch_data()
    anet = NeuralNetwork() 
    anet.fit(RBUF)
    #anet.save_model('./models/AllData.h5')


def fetch_data():
    RBUF = []
    list_of_data = ['RBUF_game_cases.pkl']
    player = pickle.load(open(list_of_data[0], "rb"))
    
    for i in range(1,len(list_of_data)):
        board_state = pickle.load(open(list_of_data[i], "rb"))
        RBUF = np.concatenate((player,board_state))

    return RBUF
# previous best 300
#train_all_data()



#Lage et supervised set 
#Kjør 500 simuleringer med MCTS  med random rollout 
#Lagre til disk og teste med ulike neural net konfigurasjoner 
#Starte ikke treningen fra skratsj 

"""By pregenerating a dataset through simulations 
and testing different neural network configurations
 on this dataset before the actual training begins, 
 you can potentially accelerate the learning process 
 of the neural network once it starts training on 
 the entire dataset. This approach aims to provide
a head startby providing the neural network with relevant 
training data and optimizing its architecture before diving 
into the full training process."""
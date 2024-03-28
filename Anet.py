import tensorflow as tf
from Hex import hex
from numpy import np
import random

print("TensorFlow version:", tf.__version__)

class NeuralNetwork:
    def __init__(self, activation_function, hidden_layers, learning_rate, optimizer, epochs, board_size): 
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.model = self.build_model()
        self.hidden_layers = hidden_layers
        self.optmizer = optimizer
        self.epochs = epochs
        self.board_size=board_size
        self.epsilon = 1
        #input size
        #output size
    
    #Builds neural network
    def build_model(self):
        model = tf.keras.Sequential() #groups a linear stack of layers into a tf.keras.Model.
        model.add(tf.keras.Input(shape=(self.board_size**2+1,))) #Input layer. Board state + PID.
        for neurons in self.hidden_layers:
            model.add(tf.keras.Dense(neurons, activation=self.activation_function,)) #Hidden layers
        model.add(tf.keras.Dense(self.board_size**2, activation=tf.keras.softmax,)) #Output layer. Må vi ha softmax?

        #We specify the training configuration (optimizer, loss, metrics):
        model.compile(optimizer = self.optimizer(learning_rate=self.learning_rate), loss = ['mse'], metrics = ['accuracy'])
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
        board_state = np.array(game).flatten()
        board_state = np.insert(board_state, 0, game.player_turn) #hvordan vet man at den sjekker PID i tening og predict?
        output = self.model.predict(board_state).numpy().flatten() 
        #Need to check valid moves
        for i in range(len(valid_and_invalid_actions)): #if action is invalid, set output to 0
            if valid_and_invalid_actions[i] == 0:
                output[i] == 0
        output = tf.nn.softmax(output)
        max_index = np.argmax(output)
        return valid_and_invalid_actions[max_index] #returns best move?
    
     
    
    #fix format

    #Function to get weigts+biases saved to file for each interval
    #dette er komplisert, trenger ikke dette (enda)!
    '''
    def select_epsilon_greedy(self, hex_state):
        self.epsilon = self.epsilon * 0.9
        if random.random() < 1: #Burde kunne endre epsilon
            return self.choose_uniform(hex_state)
        return self.select_greedy_move(hex_state)
    
    def choose_uniform(self, hex_state):
        board = hex_state[1:]
        valid_moves = [i for i in range(len(board)) if board[i] == 0]
        return random.choice(valid_moves)

    '''


    def save_model(self, path):
        self.model.save_weights(path)

    def load_model(self, path):
        self.model.load_weights(path)

    def restart_epsilon(self):
        self.epsilon = 1 #Burde kunne restarte til en annen epsilon enn 1

    def get_epsilon(self):
        return self.epsilon
    
    def shrink_epsilon(self):
        self.epsilon = self.epsilon * 0.99

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
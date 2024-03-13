import tensorflow as tf
from Hex import hex

print("TensorFlow version:", tf.__version__)

class NeuralNetwork:
    def __init__(self, state, activation_function, hidden_layers, learning_rate, optimizer): #burde vi initialize alle paramtere?
        self.learning_rate = learning_rate
        self.state = state
        self.activation_function = activation_function
        self.model = self.build_model()
        self.hidden_layers = hidden_layers
        self.optmizer = optimizer

    def build_model(self):
        model = tf.keras.Sequential() #groups a linear stack of layers into a tf.keras.Model.
        model.add(tf.keras.Input(shape=(hex.board_size**2+1,))) #Input layer. Board state + player identifier
        for neurons in self.hidden_layers:
            model.add(tf.keras.Dense(neurons, activation=self.activation_function,)) #Hidden layers
        model.add(tf.keras.Dense(hex.board_size**2, activation=tf.keras.softmax,)) #Output layer

        #We specify the training configuration (optimizer, loss, metrics):
        model.compile(optimizer = self.optimizer(learning_rate=self.learning_rate), loss = ['mse'], metrics = ['accuracy'])
        model.summary()
        return model
    



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
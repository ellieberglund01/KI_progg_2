
class NeuralNetwork: #Tensorfow eller Pychorch
    def __init__(self, state,activation_function, learning_rate):
        self.learning_rate = learning_rate
        self.state = state
        self.activation_function = activation_function



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
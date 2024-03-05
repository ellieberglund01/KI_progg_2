
class NeuralNetwork: #Tensorfow eller Pychorch
    def __init__(self, state,activation_function, learning_rate):
        self.learning_rate = learning_rate
        self.state = state
        self.activation_function = activation_function
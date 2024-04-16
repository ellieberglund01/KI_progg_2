from keras.activations import relu, softmax, linear, sigmoid, tanh
#from keras.losses import mse
from keras.optimizers import Adam, Adagrad, SGD, RMSprop


#MCTS
EXPLORATION_RATE = 0.001
NUMBER_SEARCH_GAMES = 100
TOTAL_EPISODES = 16

#Hex
SIZE = 5
VISUALIZATION = False

#Anet 
HIDDEN_LAYERS = [64, 64, 64]
TOTAL_BATCHES = 3000
TRAINING_BATCH = 64
INTERVAL = 4
ACTIVATION_FUNCTION = relu
OPTIMIZER = 'adam'
EPOCHS = 10
LEARNING_RATE = 0.001

#TOPP
TOPP_GAMES = 4



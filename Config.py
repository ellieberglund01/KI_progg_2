from keras.activations import relu, softmax, linear, sigmoid, tanh
#from keras.losses import mse
from keras.optimizers import Adam, Adagrad, SGD, RMSprop


#MCTS
EXPLORATION_RATE = 0.001
NUMBER_SEARCH_GAMES = 100
TOTAL_EPISODES = 5

#Hex
SIZE = 7
VISUALIZATION = False

#Anet 
HIDDEN_LAYERS = [128, 128, 128]
TOTAL_BATCHES = 3000
TRAINING_BATCH = 64
INTERVAL = 5
ACTIVATION_FUNCTION = relu
OPTIMIZER = Adam
EPOCHS = 10
LEARNING_RATE = 0.001

#TOPP
TOPP_GAMES = 5



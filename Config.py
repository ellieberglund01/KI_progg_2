from keras.activations import relu, softmax, linear, sigmoid, tanh
from keras.losses import categorical_crossentropy, mse
from keras.optimizers import Adam, Adagrad, SGD, RMSprop


#MCTS
EXPLORATION_RATE = 0.01
NUMBER_SEARCH_GAMES = 1000
TOTAL_EPISODES = 1000

#Hex
SIZE = 7
VISUALIZATION = False

#Anet 
HIDDEN_LAYERS = [128, 128, 128]
TOTAL_BATCHES = 3000
TRAINING_BATCH = 128
INTERVAL = 2
ACTIVATION_FUNCTION = relu
OPTIMIZER = Adam
EPOCHS = 10
LEARNING_RATE = 0.001







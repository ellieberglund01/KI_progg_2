import os
import numpy as np
import copy
import random
import pickle
from Anet import NeuralNetwork
from MCTS_new import MCTS, Node
from Hex import HexGame 
from Display import DisplayGame
from keras.optimizers import Adam
from keras.activations import relu

TRAINING_BATCH = 10
SIZE = 3
TOTAL_EPISODES = 10
TOTAL_BATCH = 10
INTERVAL = 10
HIDDEN_LAYERS = [128, 128, 128]
NUMBER_OF_BATCHES = 3000
ACTIVATION_FUNCTION = relu
OPTIMIZER = Adam
LEARNING_RATE = 0.001
EPOCHS = 3
EXPLORATION_RATE = 0.01


class ReinforcementLearner():

    def __init__(self):
        self.model_path = os.path.join(os.getcwd(), 'models') #?
        self.episode_files = []
        self.RBUF = []

    #Do we need this?
    '''
    def save_episodes_to_file(self, anet, save_interval):
        for ep in range (0,  TOTAL_EPISODES+ 1,save_interval):
            filename = f'anet_{ep}.pt'
            anet.save_model(os.path.join(self.model_path, filename))
    '''

    def reinforcement_learner(self, exploration_rate, path_to_weights=None):
        index = 0

        #Step 1 
        interval = INTERVAL #How frequently do we save the anet based on the amount of episodes run
        
        #Step 2 
        self.RBUF = []
        
        #Step 3
        ANET = NeuralNetwork(ACTIVATION_FUNCTION, HIDDEN_LAYERS, LEARNING_RATE, OPTIMIZER, EPOCHS) #initialize ANET. Define hex?

        if path_to_weights != None:
            ANET.load_model(path_to_weights) #anet function

        #Step 4
        for ep in range(TOTAL_EPISODES+1):
            ANET.restart_epilson() #Sets epsilon = 1 
            print(f"This is the {ep}. game")
            
            #(a)(b)
            hex = HexGame(n=SIZE) 
            display = DisplayGame(hex)
            #display.draw_board(hex.winning_player, "Player 1", "Player 2")
            start_node = Node(1,None,None) #Player 1 starts 
            
            #(c)
            mcts = MCTS(hex, start_node, EXPLORATION_RATE)
            print("Run MCTS to game over")
            print("-------------------------")
            

            #(d)
            while not hex.is_game_over():
                root = start_node
                index =+ 1
                selected_node, D= mcts.choose_action(start_node)
                board_state = hex.get_flat_representation()
                board_state_inc_player = np.insert(board_state, 0, hex.player_turn) 
                game_case = (board_state_inc_player, D)
                
                if len(self.RBUF) < TOTAL_BATCH:
                    self.RBUF.append(game_case) #Need to set a limit on number of batches 
                else:
                    overwritten_index = index % TOTAL_BATCH #If we have reaced limit in batches the buffer is overwritten with new data cases 
                    self.RBUF[overwritten_index] = game_case
                
                chosen_action = np.argmax(D) #Here we choose the action based on the highest distribution, check again 
                hex.move(chosen_action)
                display.draw_board(hex.winning_player, "Player 1", "Player 2")

                #Do we need this:?
                """D_copy = copy.deepcopy(D)
                while not hex.valid_move(actual_move):
                    D_copy[actual_move] = 0
                    actual_move = np.argmax(D_copy)"""
                
                for i in range(len(root.children)):
                    if root.children[i].parent_action == chosen_action:
                        new_root = root.children[i]
                        break
                
                assert new_root != root, "Not able to creat a new root" #check to ensure that new_root is different from current root
                root = new_root 
            
            #Game finished
            display.draw_board(hex.winning_player, "Player 1", "Player 2")
            print(f"DONE with MCTs for game: {ep}")

            #(e)
            if len(self.RBUF) < TRAINING_BATCH:
                train = self.RBUF

            else:
                train = random.sample(self.RBUF, TRAINING_BATCH) #Take random sample batch from the the total batch in RBUF
            
            ANET.fit(train) #anet function
            number_of_anet = 1
            #(f)
            if ep % interval == 0: #if interval= 10 this will be true for ep = 10,20, 30, 40 ....
                print("Saving anet's parameters for later use in tournament")
                number_of_anet += 1
                filename = f'anet{number_of_anet}-{ep}.h5'
                ANET.save_model(os.path.join(self.model_path, filename)) #How does this work?
                self.episode_files.append(filename)
                pickle.dump(self.RBUF, open("......")) #What is this?
        
        pickle.dump(self.RBUF, open("......"))#What is this?
    
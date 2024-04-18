import os
import numpy as np
import copy
import random
import pickle
from Anet import NeuralNetwork
from MCTS_new import MCTS, Node
from Hex import HexGame 
from Display import DisplayGame
from tensorflow import keras
from keras.optimizers import Adam
from keras.activations import relu
import tensorflow as tf
import torch
from Config import *

class ReinforcementLearner():
    def __init__(self):
        self.RBUF = []

    def reinforcement_learner(self):
        index = 0
        #Step 1 
        interval = INTERVAL #How frequently do we save parameters from anet based on the amount of episodes run. 
        #Step 2 
        self.RBUF = []
        #Step 3
        ANET = NeuralNetwork(ACTIVATION_FUNCTION, HIDDEN_LAYERS, LEARNING_RATE, OPTIMIZER, EPOCHS, SIZE) #initialize ANET. Models gets built

        filename = f'anet0.weights.h5'
        ANET.save_weights(filename)
        
        #Step 4
        for ep in range(1,TOTAL_EPISODES+1):
            print(ep)
            ANET.restart_epsilon() 
            print(f"This is the {ep}. game")
            #(a)(b)
            hex = HexGame(SIZE,None)
            display = DisplayGame(hex)
            start_node = Node(hex.player_turn,None,None) 
            mcts = MCTS(hex, start_node, EXPLORATION_RATE, ANET) 
            
            #(d)          
            while not hex.is_game_over():
                index += 1
                D1, D2 = mcts.choose_action(start_node, NUMBER_SEARCH_GAMES) 
                if random.random() < 0.2:
                    D1_normalized = np.array(D1)/sum(D1)
                    index = np.random.choice(len(D1_normalized),p=D1_normalized)
                else:
                    index = np.argmax(D1) 
                best_child =  start_node.children[index]
                selected_action = best_child.parent_action 
                print("selected action based on D", selected_action)
                
                board_state = np.array(hex.board).flatten()
                board_state_inc_player = np.insert(board_state, 0, hex.player_turn) #sets player_turn at index 0
                game_case = (board_state_inc_player, D2)
                
                if len(self.RBUF) < TOTAL_BATCHES:
                    self.RBUF.append(game_case) #Need to set a limit 
                    
                else:
                    overwritten_index = index % TOTAL_BATCHES #If we have reaced limit in batches the buffer is overwritten with new data cases 
                    self.RBUF[overwritten_index] = game_case 
                hex.move(selected_action)

                if VISUALIZATION:
                    display.draw_board(None,"player 1", "player 2")
                    hex.display()

                start_node = best_child
                mcts = MCTS(hex, start_node, EXPLORATION_RATE, ANET)
                ANET.epsilon = ANET.epsilon * 0.999

            if VISUALIZATION:
                display.draw_board(hex.player_turn, "Player 1", "Player 2")
            print(f"DONE with MCTs for game: {ep}")

            #(e)
            if len(self.RBUF) < TRAINING_BATCH:
                train = self.RBUF
            else:
                train = random.sample(self.RBUF, TRAINING_BATCH) #Take random sample batch from the the total batch in RBUF  
            
            ANET.fit(train) #train anet on the random sample batch
            print(f"Done training on episode {ep}")
            #(f)
            if ep % interval == 0: #if interval= 10 this will be true for ep = 10,20, 30, 40 ....
                print("Saving anet's parameters for later use in tournament")
                filename = f'anet{ep}.weights.h5'
                ANET.save_weights(filename)
            
        # Save RBUF using pickle
        with open('RBUF_game_cases3.pkl', 'wb') as f:
            pickle.dump(self.RBUF, f)

RL = ReinforcementLearner()
RL.reinforcement_learner()







import numpy as np
import random
import pickle
from Anet import NeuralNetwork
from MCTS import MCTS, Node
from Hex import HexGame 
from Display import DisplayGame
from Config import *

class ReinforcementLearner():
    def __init__(self):
        self.RBUF = []

    def reinforcement_learner(self):
        index = 0
        interval = INTERVAL #step1
        self.RBUF = [] #step2
        ANET = NeuralNetwork(ACTIVATION_FUNCTION, HIDDEN_LAYERS, LEARNING_RATE, OPTIMIZER, EPOCHS, SIZE) #step3

        filename = f'anets/anetsgd_0.weights.h5'
        ANET.save_weights(filename)
        
        #Step 4
        for ep in range(1,TOTAL_EPISODES+1):
            ANET.restart_epsilon() #Epsilon is set to 1 after each game
            print(f"This is the {ep}. game")
            hex = HexGame(SIZE) #a,b
            start_node = Node(hex.player_turn) 
            mcts = MCTS(hex, start_node, EXPLORATION_RATE, ANET) #c
            display = DisplayGame(hex)

            #d          
            while not hex.is_game_over():
                index += 1
                D_ex, D_in = mcts.choose_action(start_node, NUMBER_SEARCH_GAMES) 

                #add game case to rbuf
                board_state = np.array(hex.board).flatten()
                board_state_inc_player = np.insert(board_state, 0, hex.player_turn) #sets player_turn at index 0
                game_case = (board_state_inc_player, D_in)
                
                if len(self.RBUF) < TOTAL_BATCHES: #Append game case to rbuf
                    self.RBUF.append(game_case) 
                else:
                    overwritten_index = index % TOTAL_BATCHES #If we have reaced limit in batches the buffer is overwritten with new gamecases 
                    self.RBUF[overwritten_index] = game_case 
                
                #choose actual move based on D
                if random.random() < 0.2:
                    index = np.random.choice(len(D_ex),p=D_ex)
                else:
                    index = np.argmax(D_ex) 
                
                best_child =  start_node.children[index]
                selected_action = best_child.parent_action 
                print("selected action based on D", selected_action)
                hex.move(selected_action)
          
                if VISUALIZATION:
                    display.draw_board(None,"player 1", "player 2")
                    hex.display()

                #retain subtree, discard everything else
                start_node = best_child
                mcts.root_node = start_node
                mcts.root_node.parent = None 
                ANET.epsilon = ANET.epsilon * 0.999

            if VISUALIZATION:
                display.draw_board(hex.player_turn, "Player 1", "Player 2")
            print(f"DONE with MCTs for game: {ep}")

            #e: train anet on the random sample batch
            if len(self.RBUF) < TRAINING_BATCH:
                train = self.RBUF
            else:
                train = random.sample(self.RBUF, TRAINING_BATCH) #Take random sample batch from the the total batch in RBUF  
            ANET.fit(train) 
            print(f"Done training on episode {ep}")
       
            #f
            if ep % interval == 0: 
                print("Saving anet's parameters for later use in tournament")
                filename = f'anets/anetsgd_{ep}.weights.h5'
                ANET.save_weights(filename)
             
        # Save RBUF using pickle: used for tuning
        with open('RBUF_game_cases3.pkl', 'wb') as f:
            pickle.dump(self.RBUF, f)


RL = ReinforcementLearner()
RL.reinforcement_learner()







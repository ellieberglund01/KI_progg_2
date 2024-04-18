from Anet import NeuralNetwork
from Config import *
from Hex import HexGame 
import numpy as np
from Display import DisplayGame
import random
from MCTS_new import MCTS, Node
import matplotlib.pyplot as plt


class TOPP:
    def __init__(self, n_games):
        self.n_games = n_games 
        self.points_per_anet = {}
        self.scores_per_series = {}
        self.anet_episodes = {}

    def load_agents_5(self):
        agents = []
        EP = 240
        I = 40
        for ep in range(EP+1):
            if ep % I == 0:
                anet = NeuralNetwork(ACTIVATION_FUNCTION, HIDDEN_LAYERS, LEARNING_RATE, OPTIMIZER, EPOCHS, 5)
                filename = f'anet{ep}.weights.h5'
                anet.load_weights(filename)
                agents.append(anet)
                self.points_per_anet[anet]= 0
                self.anet_episodes[anet] = ep
        return agents 
    
    def load_agents_7(self):
        agents = []
        EP = 40
        I = 40
        for ep in range(EP+1):
            if ep % I == 0:
                anet = NeuralNetwork(ACTIVATION_FUNCTION, HIDDEN_LAYERS, LEARNING_RATE, OPTIMIZER, EPOCHS, 7)
                filename = f'anet7_{ep}.weights.h5'
                anet.load_weights(filename)
                agents.append(anet)
                self.points_per_anet[anet]= 0
                self.anet_episodes[anet] = ep
        return agents 

    def play_game(self, agent1, agent2):
        hex = HexGame(SIZE) 
        #display = DisplayGame(hex)
        while not hex.is_game_over():
            actions = hex.get_legal_actions_with_0()
            board_state = np.array(hex.board).flatten()
            board_state = np.insert(board_state, 0, hex.player_turn)
            #display.draw_board(None,"player 1", "player 2")
            #hex.display()
            if hex.player_turn == 1:
                action = agent1.select_best_move_random(actions,hex) #argmax eller select best move random?
                hex.move(action)
            else:
                action = agent2.select_best_move_random(actions, hex)
                hex.move(action) 
        winner = hex.game_result()                
        print('GAME OVER')
        print('Winner:', winner)
        #display.draw_board(hex.player_turn, "Player 1", "Player 2")
        #hex.display()
        if winner == 1:
            self.points_per_anet[agent1] += 1
        else:
            self.points_per_anet[agent2] += 1
        return agent1 if hex.winner_player == 1 else agent2
    
    def run_tournament(self, agents): #change to player
        for i, player1 in enumerate(agents):
            for j, player2 in enumerate(agents):
                if i < j:
                    series = [(player1, player2) for _ in range(self.n_games)]
                    player1_wins, player2_wins = 0, 0
                    for game in series:
                        winner = self.play_game(game[0], game[1])
                        if winner == player1:
                            player1_wins += 1
                        elif winner == player2:
                            player2_wins += 1
                    self.scores_per_series[(player1, player2)] = (player1_wins, player2_wins)
            
    #Show which anet is playing which agent
    def display_results(self):
        for matchup, score in self.scores_per_series.items():
            print(f'Anet{self.anet_episodes[matchup[0]]} vs Anet{self.anet_episodes[matchup[1]]}: {score[0]} - {score[1]}')
        for anet, points in self.points_per_anet.items():
            print(f'Anet{self.anet_episodes[anet]}: {points} points')
        # Extracting data from dictionaries
        anet_episodes = [self.anet_episodes[anet] for anet in self.points_per_anet.keys()]
        points = list(self.points_per_anet.values())

        # Plotting
        plt.bar(anet_episodes, points, color='blue')
        plt.xlabel('Anet')
        plt.ylabel('Points')
        plt.title('Points per Anet')
        plt.xticks(anet_episodes)
        plt.show()    

    def play_game_MCTS_random(self):
        game_result_MCTS = 0
        game_result_random = 0
        for i in range(20):
            hex = HexGame(3,2)
            ANET = NeuralNetwork(ACTIVATION_FUNCTION, HIDDEN_LAYERS, LEARNING_RATE, OPTIMIZER, EPOCHS, SIZE)
            #display = DisplayGame(hex)

            while not hex.is_game_over():
                legal_actions = hex.get_legal_actions()
                board_state = np.array(hex.board).flatten()
                board_state = np.insert(board_state, 0, hex.player_turn)
                #display.draw_board(None,"player 1", "player 2")
                
                #hex.display()
                if hex.player_turn == 1:
                    start_node = Node(1,None,None) #Player 1 starts
                    mcts_player = MCTS(hex, start_node, EXPLORATION_RATE, ANET)
                    D1, D2 = mcts_player.choose_action(start_node, NUMBER_SEARCH_GAMES) 
                    best_child_index = np.argmax(D1)
                    best_child =  start_node.children[best_child_index]
                    action = best_child.parent_action 
                    hex.move(action)
                    #start_node = best_child
                    #mcts_player = MCTS(hex, start_node, EXPLORATION_RATE, ANET)
                else:
                    action = random.choice(legal_actions)
                    hex.move(action) 

            winner = hex.game_result()
            if winner == 1:
                game_result_MCTS += 1
            else:
                game_result_random += 1      
            #print('GAME OVER')
            #print('Winner:', winner)
            #display.draw_board(hex.player_turn, "Player 1", "Player 2")
            #hex.display()
            print("mcts", game_result_MCTS)
            print("random", game_result_random)
              

topp = TOPP(n_games=TOPP_GAMES)
agents = topp.load_agents_7()
topp.run_tournament(agents)
topp.display_results()

random.seed(4)
np.random.seed(4)
#topp.play_game_MCTS_random()
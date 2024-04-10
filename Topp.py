from Anet import NeuralNetwork
from Config import *
from Hex import HexGame 
import numpy as np
from Display import DisplayGame
import random


class TOPP:
    def __init__(self, n_games):
        self.n_games = n_games 
        self.points_per_anet = {}
        self.scores_per_series = {}
        self.policies = ['anet0.weights.h5','anet5.weights.h5']

    def load_agents2(self):
        agents = []
        for ep in range(1,TOTAL_EPISODES+1):
            if ep % INTERVAL == 0:
                anet = NeuralNetwork(ACTIVATION_FUNCTION, HIDDEN_LAYERS, LEARNING_RATE, OPTIMIZER, EPOCHS, SIZE)
                filename = f'anet{ep}.weights.h5'
                anet.load_weights(filename)
                agents.append(anet)
                self.points_per_anet[anet]= 0
        return agents 
    
    def load_agents(self):
        agents = []
        for filename in self.policies:
            anet = NeuralNetwork(ACTIVATION_FUNCTION, HIDDEN_LAYERS, LEARNING_RATE, OPTIMIZER, EPOCHS, SIZE)
            anet.load_weights(filename)
            agents.append(anet)
            self.points_per_anet[anet]= 0
        return agents 
    

    def play_game(self, agent1, agent2, player_to_start):
        # Implement the game logic here and return the winner
        hex = HexGame(SIZE)
        display = DisplayGame(hex)
        while not hex.is_game_over():
            actions = hex.get_legal_actions_with_0()
            board_state = np.array(hex.board).flatten()
            board_state = np.insert(board_state, 0, hex.player_turn)
            display.draw_board(None,"player 1", "player 2")
            hex.display()
            if player_to_start == 1:
                action = agent1.predict(actions,hex)
                hex.move(action)
            else:
                action = agent2.predict(actions, hex)
                hex.move(action)    
        print('GAME OVER')
        display.draw_board(hex.player_turn, "Player 1", "Player 2")
        hex.display()
        if hex.winner_player == 1:
            self.points_per_anet[agent1] += 1
        else:
            self.points_per_anet[agent2] += 1
        
        return agent1 if hex.winner_player == 1 else agent2
    
    def run_tournament(self, agents):
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i < j:
                    series = [(agent1, agent2) for _ in range(self.n_games)]
                    agent1_wins, agent2_wins = 0, 0
                    for game in series:
                        start_player = random.randint(1, 2)
                        winner = self.play_game(game[0], game[1], start_player)
                        if winner == agent1:
                            agent1_wins += 1
                        elif winner == agent2:
                            agent2_wins += 1
                    self.scores_per_series[(i, j)] = (agent1_wins, agent2_wins)

    def display_results(self):
        for matchup, score in self.scores_per_series.items():
            print(f'Agent {matchup[0]+1} vs Agent {matchup[1]+1}: {score[0]} - {score[1]}')

topp = TOPP(n_games=TOPP_GAMES)
agents = topp.load_agents2()
topp.run_tournament(agents)
topp.display_results()



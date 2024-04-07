import os
from collections import deque
from itertools import combinations
import time
from Anet import NeuralNetwork
from Config import *
from Hex import HexGame 
import numpy as np
from Display import DisplayGame
import random


class TOPP:
    def __init__(self, n_games, series=10):
        self.n_games = n_games 
        self.series = series
        self.points = {}
      
        self.policies = ['anet0.weights.h5','anet5.weights.h5']
        self.scores = {}

    def load_agents(self):
        agents = []
        for filename in self.policies:
            anet = NeuralNetwork(ACTIVATION_FUNCTION, HIDDEN_LAYERS, LEARNING_RATE, OPTIMIZER, EPOCHS, SIZE)
            anet.load_weights(filename)
            agents.append(anet)
            self.points[anet] = 0
        return agents

    def play_game(self, agent1, agent2, player_to_start):
        # Implement the game logic here and return the winner
        hex = HexGame(SIZE)
        display = DisplayGame(hex)
        while not hex.is_game_over():
            actions = hex.get_legal_actions_with_0()
            board_state = np.array(hex.board).flatten()
            board_state = np.insert(board_state, 0, hex.player_turn)
            display.draw_board(hex.player_turn, "Player 1", "Player 2")
            if player_to_start == 1:
                move = agent1.predict(actions,board_state) #lag ny funksjon
                if hex.is_valid_move(move):
                    hex.make_move(move)
            else:
                move = agent2.predict(actions, board_state)
                if hex.valid_move(move):
                    hex.make_move(move)
            
        if hex.winning_player == 1:
            self.points[agent1] += 1
        else:
            self.points[agent2] += 1
        
        return agent1 if hex.winning_player == 1 else agent2
    
    def run_tournament(self, agents):
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i < j:
                    series = [(agent1, agent2) for _ in range(self.n_games)]
                    agent1_wins, agent2_wins = 0, 0
                    for game in series:
                        start_player = random.randint(1, 2)
                        winner = self.play_game(*game, start_player)
                        if winner == agent1:
                            agent1_wins += 1
                        elif winner == agent2:
                            agent2_wins += 1
                    self.scores[(i, j)] = (agent1_wins, agent2_wins)

    def display_results(self):
        for matchup, score in self.scores.items():
            print(f'Agent {matchup[0]} vs Agent {matchup[1]}: {score[0]} - {score[1]}')
        #print leaderboard


topp = TOPP(n_games=TOPP_G)
agents = topp.load_agents()
topp.play_game(agents[0], agents[1], 1)


#
## Run the tournament
#topp.run_tournament(agents)
#
## Display the results
#topp.display_results()
#
## Print out the leaderboard
#a = 0
#for i in topp.points:
#   print(a, topp.points[i])
#   a += 1

#play_game()
from Nim import NimGame
from MCTS import Node

game1 = NimGame(10, 3, 1)
mcts = Node(state=game1)
mcts.choose_action()
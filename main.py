from Nim import NimGame
from MCTS import Node

game = NimGame(10, 3, 1)
mcts_player1 = Node(state=game)
mcts_player2 = Node(state=game)

while not game.is_game_over():
        print("Player 1's turn:")
        selected_node = mcts_player1.choose_action()
        action = selected_node.parent_action
        print(game.N, game.K, game.current_pieces) #Må på en måte initialisere gamet på nytt (siden vi spiller jo ferdig gamet når vi simulerer)
        print(action)
        game.move(action)
        print(f"Player 1 takes {action} piece(s).")
        game.display()

        if game.is_game_over():
            break

        print("Player 2's turn:")
        selected_node = mcts_player2.choose_action()
        action = selected_node.parent_action
        game.move(action)
        print(f"Player 2 takes {action} piece(s).")
        game.display()


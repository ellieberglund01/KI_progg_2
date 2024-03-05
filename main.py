from Nim import NimGame
from MCTS import Node


def run_nim_game():
    game = NimGame(10, 3, 1)
    mcts_player1 = Node(state=game)
    mcts_player2 = Node(state=game)

    while not game.is_game_over():
        #Må på en måte initialisere gamet på nytt (siden vi spiller jo ferdig gamet når vi simulerer)

        # Display the current game state before each player's turn
        print(f"\nCurrent game state: {game.current_pieces} pieces remaining")

        print("Player 1's turn:")
        mcts_player1.state = game  # Ensure the MCTS node's state is up to date
        selected_node = mcts_player1.choose_action()
        action = selected_node.parent_action
        print(f"Player 1 takes {action} piece(s).")
        print("------------------------------------------")
        game.move(action)
        game.display()
        if game.is_game_over():
            break

        print("Player 2's turn:")
        mcts_player2.state = game  # Ensure the MCTS node's state is up to date

        selected_node = mcts_player2.choose_action()
        action = selected_node.parent_action
        game.move(action)
        print(f"Player 2 takes {action} piece(s).")
        game.display()

if __name__ == "__main__":
    run_nim_game()

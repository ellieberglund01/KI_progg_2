from Nim import NimGame
from MCTS_new import Node
from MCTS_new import MCTS
from Hex import HexGame


def run_game():
    #game = NimGame(4, 3, 1)
    game = HexGame(3)
    start_node = Node(1,None,None)
    mcts_game = MCTS(game,start_node,0.01)

    
    while not game.is_game_over():
        selected_node = mcts_game.choose_action(start_node)
        selected_action = selected_node.parent_action 
        print("-----------------------------")
        print("Player", game.player_turn, "places a piece on", {selected_action})
        print("-----------------------------")
        game.move(selected_action)
        
        game.display()

        if game.is_game_over():
            break
        start_node = selected_node
        mcts_game = MCTS(game, start_node,1)

    print("GAME OVERRRRRER")

if __name__ == "__main__":
    run_game()

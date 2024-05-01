from Nim import NimGame
from MCTS import Node
from MCTS import MCTS
from Hex import HexGame
from Display import DisplayGame


def run_game():
    #game = NimGame(4, 3, 1)
    game = HexGame(3)
    start_node = Node(1,None,None)
    mcts_game = MCTS(game,start_node,1)
    display = DisplayGame(game)
    
    while not game.is_game_over():
        selected_node, D = mcts_game.choose_action(start_node)
        selected_action = selected_node.parent_action 
        print("-----------------------------")
        print("Player", game.player_turn, "places a piece on", {selected_action})
        print("-----------------------------")
        game.move(selected_action)
        
        display.draw_board(None,"player 1", "player 2")
        game.display()

        if game.is_game_over():
            break
        start_node = selected_node
        mcts_game = MCTS(game, start_node,1)
    
    display.draw_board(game.player_turn,"player 1", "player 2")
    print("GAME OVERRRRRER")

if __name__ == "__main__":
    run_game()

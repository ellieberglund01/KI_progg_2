from Nim import NimGame
from MCTScopy import Node
from MCTScopy import MCTS


def run_nim_game():
    game = NimGame(10, 3, 1)
    start_node = Node(1,None,None)
    mcts_game = MCTS(game,start_node,1)

    
    while not game.is_game_over():
        print(f"\nCurrent game state: {game.current_pieces} pieces remaining")
        selected_node = mcts_game.choose_action(start_node)
        selected_action = selected_node.parent_action 
        print("------------------------------------------")
        print(f"Game Player {game.player_turn} takes action {selected_action}") #Denne oppdateres ikke riktig etter første simulering
        #print(f"Node Player {selected_node.parent.player} takes action {selected_node.parent_action}")
        game.move(selected_action)
        
        game.display()

        if game.is_game_over():
            break
        start_node = selected_node
        mcts_game = MCTS(game, start_node,1)
        #print(f"New Game Player {game.player_turn}")
        #print(f"New Root node player:{start_node.player}")

  

if __name__ == "__main__":
    run_nim_game()

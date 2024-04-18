# Import and initialize your own actor
from Anet import NeuralNetwork
from Config import *



# Import and override the `handle_get_action` hook in ActorClient
from ActorClient import ActorClient
from Hex import HexGame

class MyClient(ActorClient):
    def __init__(self):
        self.hex = HexGame(SIZE)
        actor = NeuralNetwork(ACTIVATION_FUNCTION, HIDDEN_LAYERS, LEARNING_RATE, OPTIMIZER, EPOCHS, SIZE)
        filename = f'anet7_0.weights.h5'
        actor.load_weights(filename)
        self.actor = actor
        super().__init__()

    def transform_board(self,state):
       board = []
       board_row = []
       for i in range(1,len(state)):
            if state[i] == 0:
                board_row.append((0,0))
            elif state[i] == 1:
                board_row.append((1,0))
            else:
                board_row.append((0,1))

            if i % SIZE == 0:
                board.append(board_row)
                board_row = []
       self.hex.board = board
       self.hex.player_turn = state[0]
       #Gjør om fra liste til tupler  
       
    #Hvordan få vår logikk til å passe med handle_get_action?
    def handle_get_action(self, state): #change state to tuple. Er state et hex object? Endre egen board state eller state
        self.transform_board(state)
        valid_and_invalid_actions = self.hex.get_legal_actions_with_0() #er state et hex object?
        action = self.actor.predict(valid_and_invalid_actions, self.hex) # Your logic
        row = action[0]
        col = action[1]
        return int(row), int(col)

# Initialize and run your overridden client when the script is executed
if __name__ == '__main__':
 client = MyClient()
 client.run()

 #ANNET:
 #state_manager. Ikke definert som egen fil i oppgave, men blir nevnt. Er det viktig?
 #Anet performer ikke sånn superbra :)) Hvordan kan vi validere dette? 
 #I topp, skal vi kjøre predict eller random? Hvilken balanse mellom predict/random er lurt? Nå har vi 0.2
 #Få de til å sjekke display av board + winning path 
 #
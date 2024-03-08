class NimGame:
    def __init__(self, N, K, player_turn):
        self.N = N  # Total number of pieces on the board
        self.K = K  # Maximum number of pieces a player can take in one turn
        self.current_pieces = N  # Current number of pieces on the board
        self.player_turn = player_turn #1 if player 1, -1 for player 2
    
    def get_legal_actions(self):
        """Return a list of the legal actions based on the current state."""
        return list(range(1, min(self.K, self.current_pieces) + 1))
    
    def move(self, action):
        """Make a move (action), updating the state of the game."""
        print("Player:", self.player_turn)

        if action in self.get_legal_actions():  
            self.current_pieces -= action
            print("Current pieces on the board", self.current_pieces)

            if self.player_turn == 1:
                self.player_turn = -1
            else:
                self.player_turn = 1
      
        else:
            print("legal action is: ", self.get_legal_actions())
            raise ValueError("Illegal action.")
        return self
    
    
    def is_game_over(self):
        """Check if the game has reached a terminal state."""
        return self.current_pieces == 0
    
    def game_result(self):
        """
        Return the game result from the perspective of the current player.
        If the game is over, return 1 if the player has won and -1 if lost.
        """
        if not self.is_game_over():
            raise ValueError("Game is not yet finished.")
        # Assuming player 1's turn is True and player 2's turn is False
        # If it's player 1's turn in a terminal state, player 2 has taken the last piece, and vice versa.
        if self.player_turn == 1:
            print("player 1 won")
            return -1
        else:
            print("player 2 won")
            return 1

    def display(self):
        """Display the current state of the game."""
        print(f"Current number of pieces: {self.current_pieces}")



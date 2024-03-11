

class HexGame():
    def __init__(self,board_size):
        self.board_size = board_size #Number of rows (or column) of the square array 
        self.board = [[(0, 0) for _ in range(board_size)] for _ in range(board_size)] #empty cells
        self.player_turn = 1  # Player 1 starts the game

    def get_flat_representation(self):
        flat_board = [cell for row in self.board for cell in row]
        return flat_board
    
    def is_valid_move(self, row, col):
        return 0 <= row < self.board_size and 0 <= col < self.board_size and self.board[row][col] == (0,0)
    
    def get_legal_actions(self):
        legal_actions_list = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.is_valid_move(row, col):
                    legal_actions_list.append((row, col))
        return legal_actions_list

    def move(self, action):
        row = action[0]
        col = action[1]
        if action in self.get_legal_actions():
            self.board[row][col] = (1, 0) if self.player_turn == 1 else (0, 1)
            print("move made by:", {self.player_turn})
            if self.is_game_over():
                print("Player won:", self.player_turn)
            else:
                print("still playing")
            self.player_turn = 3 - self.player_turn  # Switch player
            self.display() 
        else:
            print("Not a valid move")
        return self

    def game_result(self):
        if not self.is_game_over():
            raise ValueError("Game is not yet finished.")
        if self.player_turn == 1:
            print("player 1 won")
            return -1
        else:
            print("player 2 won")
            return 1
                 
    def display(self):
        for row in self.board:
            print(" ".join(str(cell) for cell in row))
        print()

    def check_winning_condition(self, player_id):
        if player_id == 1:
            # Check entire first column for player 1's pieces
            for row in range(self.board_size):
                if self.board[row][0] == (1,0) and self.dfs((1,0), row, 0, set()):
                    return True

        elif player_id == 2:
            # Check entire first row for player 2's pieces
            for col in range(self.board_size):
                if self.board[0][col] == (0,1) and self.dfs((0,1), 0, col, set()):
                    return True       
        return False
    
    def is_game_over(self):
        return self.check_winning_condition(self.player_turn)
    

    def dfs(self, player_piece, row, col, visited):
        if player_piece == (1,0) and col == self.board_size - 1:
            return True
        if player_piece == (0,1) and row == self.board_size - 1:
            return True
        
        #Hjørner 
        if row == 0 and col == 0:
            neighbors = [(0,1), (1,0)]
        elif row == self.board_size -1 and col == 0:
            neighbors = [(-1,0), (0,1), (-1,1)]
        elif row == 0 and col == self.board_size -1:
            neighbor = [(-1,0),(1,0), (1,-1)]
        elif row == self.board_size -1 and col == self.board_size -1:
            neighbors = [(-1,0), (0,-1)]
        
        #edges
        elif row == 0:
            neighbors = [(0,-1), (1,0), (0,1), (1,-1)]
        elif col == 0:
            neighbors = [(0,1), (-1,0), (1,0), (-1,1)]
        elif row == self.board_size -1:
            neighbors = [(-1,0), (0,-1),(0,1), (-1,1)]
        elif col == self.board_size -1:
            neighbors = [(1,0), (-1,0), (0,-1), (1,-1)]
        
        #the rest 
        else:
            neighbors = [(1,0), (-1,0),(0,1),(0,-1), (1,-1), (-1,1)]
   
        #funker ikke for (0,x) (x,board_size-1), (board_size-1,x) og (x,0):
        #neighbors = [(0, 1), (-1, 1)] if player_piece == (1,0) else [(1, 0), (1, -1)]
        for neighbor in neighbors:
            nr, nc = row + neighbor[0], col + neighbor[1]
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size and (nr, nc) not in visited and self.board[nr][nc] == player_piece:
                visited.add((nr, nc))
                if self.dfs(player_piece, nr, nc, visited):
                    return True
        return False
    

game = HexGame(5)  # Create a Hex game with a 5x5 board

#example player 1 wins 
"""game.move((0,0))
game.move((2,3))  
game.move((0,1))
game.move((3,4))
game.move((0,2))
game.move((1,2))
game.move((0,3))
game.move((2,0))
game.move((0,4))"""

#example player 2 wins 
"""game.move((0,0))
game.move((0,3))  
game.move((0,1))
game.move((1,3))
game.move((1,1))
game.move((2,3))
game.move((0,2))
game.move((3,3))
game.move((0,4))
game.move((4,3))"""


#Another example where player 1 wins 
game.move((0,0))
game.move((0,0))
game.move((1,0))
game.move((5,5))
game.move((0,1))
game.move((0,3))
game.move((1,1))
game.move((4,0))
game.move((1,2))
game.move((3,4))
game.move((2,2))
game.move((1,4))
game.move((2,3))
game.move((0,2))
game.move((2,4))

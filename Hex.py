

class Grid():
    def __init__(self,board_size):
        self.board_size = board_size #Number of rows (or column) of the square array 
        self.board = [[(0, 0) for _ in range(board_size)] for _ in range(board_size)] #empty cells
        self.current_player = 1  # Player 1 starts the game

    def get_flat_representation(self):
        flat_board = [cell for row in self.board for cell in row]
        return flat_board
    
    def is_valid_move(self, row, col):
        return 0 <= row < self.board_size and 0 <= col < self.board_size and self.board[row][col] == (0,0)
    
    def move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row][col] = (1, 0) if self.current_player == 1 else (0, 1)
            print("move made by:", {self.current_player})
            if self.check_winning_condition(self.current_player):
                print("Player won:", self.current_player)
            else:
                print("still playing")
            self.current_player = 3 - self.current_player  # Switch player
            self.print_board() 
        else:
            print("Not a valid move")

            
    def print_board(self):
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

    def dfs(self, player_piece, row, col, visited):
        if player_piece == (1,0) and col == self.board_size - 1:
            return True
        if player_piece == (0,1) and row == self.board_size - 1:
            return True

        if player_piece == (1,0):
            if row == 0:
                neighbors = [(0,1)]
            else: 
                neighbors = [(0, 1), (-1, 1)]

        if player_piece == (0,1):
            if col == 0:
                neighbors = [(1,0)]
            else:
                neighbors = [(1, 0), (1, -1)]
            
        #funker ikke for (0,x) (x,board_size-1), (board_size-1,x) og (x,0)
        #neighbors = [(0, 1), (-1, 1)] if player_piece == (1,0) else [(1, 0), (1, -1)]

        for neighbor in neighbors:
            nr, nc = row + neighbor[0], col + neighbor[1]
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size and (nr, nc) not in visited and self.board[nr][nc] == player_piece:
                visited.add((nr, nc))
                if self.dfs(player_piece, nr, nc, visited):
                    return True
        return False
    




game = Grid(5)  # Create a Hex game with a 5x5 board

#example player 1 wins 
"""game.move(0,0) 
game.move(2,3)  
game.move(0,1) 
game.move(3,4)
game.move(0,2)
game.move(1,2)
game.move(0,3)
game.move(2,0)
game.move(0,4)"""


#example player 2 wins 
game.move(0,0) 
game.move(0,3)  
game.move(0,1) 
game.move(1,3)
game.move(1,1)
game.move(2,3)
game.move(0,2)
game.move(3,3)
game.move(0,4)
game.move(4,3)



"""def get_cell(self, row, col):
        if self.is_valid_cell(row, col):
            return self.board[row][col]
        return None
    
    def place_piece(self, row, col, player_id):
        cell = self.get_cell(row, col)
        if cell and not cell.occupied:
            piece = Piece(cell, player_id)
            cell.occupied = True
            cell.piece = piece
            self.player_pieces[player_id].append(piece)
            return True
        return False
    
class Cell(): 
    def __init__(self,neighbor_list, row, column):
        self.neighbor_list = neighbor_list #Up to 6 neighbors 
        self.occupied = False
        self.piece = None #If occupied is true then piece is set 
        self.row = row
        self.column = column

class Piece():
    def __init__(self,board_location, player_id):
        self.board_location= board_location
        self.player_id = player_id #red for player 1 or black for player -1"""



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
            self.current_player = 3 - self.current_player  # Switch player
            if self.check_winning_condition(self.current_player):
                print("Player won:", self.current_player)

            self.print_board() 
        
            
    def print_board(self):
        for row in self.board:
            print(" ".join(str(cell) for cell in row))
        print()

    def check_winning_condition(self, player_id):
        if player_id == 1:
            # Check entire first column for player 1's pieces
            #Noe feil her 
            for col in range(self.board_size):
                if self.board[0][col] == player_id and self.dfs(player_id, 0, col, set()):
                    return True
                
        elif player_id == 2:
            # Check entire first row for player 2's pieces
            for row in range(self.board_size):
                if self.board[row][0] == player_id and self.dfs(player_id, row, 0, set()):
                    return True
        return False

    def dfs(self, player_id, row, col, side, visited):
        if player_id == 1 and col == self.board_size - 1:
            return True
        if player_id == 2 and row == self.board_size - 1:
            return True

        neighbors = [(0, 1), (-1, 1)] if player_id == 1 else [(1, 0), (1, -1)]

        for neighbor in neighbors:
            nr, nc = row + neighbor[0], col + neighbor[1]
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size and (nr, nc) not in visited and self.board[nr][nc] == player_id:
                visited.add((nr, nc))
                if self.dfs(player_id, nr, nc, visited):
                    return True
        return False
    


# Example usage:
game = Grid(5)  # Create a Hex game with a 5x5 board
# Sequence of moves to make Player 1 win
moves = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]


game.move(0,0) 
game.move(2,3)  
game.move(1,1) 
game.move(3,4)
game.move(2,2)
game.move(1,2)
game.move(3,3)
game.move(2,0)
game.move(4,4)
print(game.check_winning_condition(1))




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

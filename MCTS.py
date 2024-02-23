import numpy as np



class Node:
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children=[]
        self.results = dict() #Definere dictionary for å holde resultater
        self.results[-1]=0
        self.results[1]=0
        self.visits = 0
        self.untried_actions = state.get_legal_actions() #get legal actions fra state class - stack
        #self.value = 0

    def is_terminal_node(self): #sjekker game over
        return self.state.is_game_over()
     
    def if_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def search(self):
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()

        return current_node 
    
    def expand(self): 
        action = self.untried_actions.pop() #kaller på random action
        next_state = self.state.move(action)
        child_node = Node(next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    def best_child(self, c_param=1): #Velger child noden med høyest verdi av Q+UCT
        q = [(c.results[1]-c.results[-1])/c.visits for c in self.children]
        u = [c_param * np.sqrt((np.log(self.visits)/(1+c.visits))) for c in self.children]
        child_values = [q[i] + u[i] for i in range(len(self.children))]  
        return self.children[np.argmax(child_values)] 
    
    def rollout(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))] # her skal ANET inn (ny klasse)


    def choose_action(self):
        simulation_no = 100 #definere i init
        for i in range(simulation_no):
            leaf = self.search()
            game_result = leaf.rollout()
            #v.backpropagate(game_result)
        
        return self.best_child(c_param=0.)
            


class MCTS(self, board, player):
    def __init__(self, board, player):
        self.board = board
        self.player = player
        self.opponent = 1 if player == 2 else 2
        self.root = Node(board, player)
    
    
    def selection(self, state, ):
        u
    def search(self):
        pass
    def expand(self):
        pass
    def rollout(self):
        pass
    def backpropagation(self):
        pass







    
class Hex:
    def __init__(self, initial_states, board_states, moves, final_states, winner):
        self.initial_states = initial_states
        self.board_states = board_states
        self.moves = moves
        self.final_states = final_states
        self.winner = winner
        self.board = Board(board_states, moves, final_states, winner)
        self.mtcs = MTCS(self.board, 1)
    


class RL_system:
    def __init__(self):
        pass
import numpy as np
from Nim import NimGame

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
     
    def is_fully_expanded(self):
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
        print("expand action: ", action)
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
        current_rollout_state = self.state
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result() 
    
    def rollout_policy(self, possible_moves): # her skal ANET inn (ny klasse)
        return possible_moves[np.random.randint(len(possible_moves))]


    def choose_action(self):
        simulation_no = 10 #definere i init
        for i in range(simulation_no):
            leaf = self.search()
            game_result = leaf.rollout(self.untried_actions)
            leaf.backpropagate(game_result) 
        print("done")
        print(self.best_child().parent_action)
        return self.best_child(c_param=0.)

    
    def backpropagate(self, result):
        self.visits += 1.
        self.results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)


            
"""def choose_action_with_anet(self, state, possible_moves):
    state_tensor = self.state_to_tensor(state)  # Convert state to tensor
    action_probabilities = self.anet.predict(state_tensor)
    adjusted_probabilities = adjust_probabilities(action_probabilities, possible_moves)
    action = np.random.choice(len(adjusted_probabilities), p=adjusted_probabilities)
    return action"""



import numpy as np
import copy 
from math import sqrt, log,inf
import numpy as np
import random

class MCTS():
    def __init__(self, game_state, root_node, exploration_rate):
        self.game = game_state
        self.root_node = root_node
        self.exploration_rate = exploration_rate 
        self.root_node.parent = None #pruner treet når en action blit tatt 

    def choose_action(self, root_node): #Policy, simulations, anet som input her også 
        print("Root node is now:", id(root_node))
        simulation_no = 10 #definere i init

        for i in range(1,simulation_no+1):
            print("Current simulation:", {i})
            print("-----------------")
            node = root_node
            game_copy = copy.deepcopy(self.game)
            leaf1, game_copy = self.tree_policy(node, game_copy) #explore best existing paths in the tree
            self.expand(leaf1, game_copy) #Adds children to leaf node
            expanded_leaf, game_copy = self.tree_policy(leaf1, game_copy) #Chooses beste action/leaf node from leaf1
            game_result = self.rollout(game_copy)  #Simulates game from expanded leaf
            path = self.backpropagate(expanded_leaf, game_result) #Backpropagates and returns the path from leaf to root
            #for node in path:
                #print("Path parent action:", node.parent_action, "Path visit count:", node.visits, "Path node value:", node.winning_count)

        #choosing the actual action in the game     
        #print("Possible actions and their values:")
        #print("current player:", root_node.player)
        best_child = max(root_node.children,key=lambda child:-child.get_value())
        #print(child_values)
        #print("For player:", root_node.player, "best action is:", best_child.parent_action)
        return best_child
    

    def rollout(self, game): #Sende inn policy/ANET også
        current_rollout_state = game
        while not current_rollout_state.is_game_over():
            possible_actions = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_actions)
            #print("Player:", game.player_turn, "takes random action",action)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result() 
    
    def rollout_policy(self, possible_moves): # her skal ANET inn (ny klasse)
        random_choice = random.choice(possible_moves)
        return random_choice 
    
    def update_visit_count(self, node):
        while node != None:
            node.visits += 1
            node = node.parent

    def backpropagate(self, node, result):
        path = []
        self.update_visit_count(node)
        while node!= None: 
            if node.player == result:
                node.sumScore += 1
            else:
                node.sumScore -= 1 
            path.append(node)
            node = node.parent
        return path
    
    def tree_policy(self, node, game):
        current_node = node
        actions = []
        #print("TREE POLICY")
        #print("number of children/actions;", len(current_node.children))
        while len(current_node.children) != 0:
            current_score = float('-inf')
            best_node = None
            for i in range (len(current_node.children)): #endre til branch?
                if current_node.children[i].get_score_for_parent(self.exploration_rate) > current_score: #if ubc to child is higher than current ubc
                    best_node = current_node.children[i]   
                    current_score = best_node.get_score_for_parent(self.exploration_rate)
                    
            current_node = best_node
            if current_node == None:
                raise Exception("No node found")
            actions.append(current_node.parent_action)
        
        for action in actions:
            game.move(action)
            #print("In tree policy, action:",action, "is taken")
        #print("----Tre policy done----")
        return current_node, game

    def create_child(self,parent, parent_action):
            if parent.player == 1: 
                child = Node(-1, parent, parent_action)
            else:
                child = Node(1, parent, parent_action)
            return child
    
    def expand(self,initial_node, game):
        actions = game.get_legal_actions()
        for action in actions:
            child = self.create_child(initial_node, action)
            if child not in initial_node.children: 
                initial_node.children.append(child)
    

class Node():
    def __init__(self, player, parent=None, parent_action=None):
        self.player = player
        self.parent = parent
        self.parent_action = parent_action
        self.children=[]
        self.visits = 0
        self.sumScore = 0 #Total wins - total losses for self.player 

    def is_terminal_node(self): 
        return self.state.is_game_over()
      
    def get_ucb_bonus(self,c):
        assert self.parent is not None, "Cannot compute ucb with none parent"
        if self.visits == 0:
            return float("inf")
        
        s_i = self.visits 
        s_p = self.parent.visits 

        return c*sqrt(np.log(s_p)/(s_i+1))
    
    def get_value(self):
        if self.visits == 0:
            return 0

        w_i = self.sumScore
        s_i = self.visits 
        return w_i/s_i

    def get_score_for_parent(self, c):
        return -self.get_value() + self.get_ucb_bonus(c)

        
        
    
    
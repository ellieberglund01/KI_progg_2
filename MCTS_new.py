import numpy as np
import copy 
from math import sqrt, log,inf
import numpy as np
import random
from Anet import NeuralNetwork

class MCTS():
    def __init__(self, game_state, root_node, exploration_rate, anet): #initialize anet
        self.game = game_state
        self.root_node = root_node
        self.exploration_rate = exploration_rate 
        self.root_node.parent = None #pruner treet når en action blit tatt 
        self.anet = anet

    def choose_action(self, root_node, n_search_games): #Policy, simulations, anet som input her også 
        print("Root node is now:", id(root_node))
        simulation_no = n_search_games #definere i init

        for i in range(1,simulation_no+1):
            print("Current simulation:", {i})
            print("-----------------")
            node = root_node
            game_copy = copy.deepcopy(self.game)
            leaf1, game_copy = self.tree_policy(node, game_copy) #explore best existing paths in the tree
            self.expand(leaf1, game_copy) #Adds children to leaf node
            expanded_leaf, game_copy = self.tree_policy(leaf1, game_copy) #Chooses beste action/leaf node from leaf1
            game_result = self.rollout(game_copy)  #Simulates game from expanded leaf
            self.backpropagate(expanded_leaf, game_result) #Backpropagates and returns the path from leaf to root
        
        #choosing the actual action in the game     
        best_child = max(root_node.children,key=lambda child:child.visits) #dette blir feil
        normalized_distribution = self.get_distribution(root_node) #denne burde kalles på i main. skal egt brukes i valg
        #print("For player:", root_node.player, "best action is:", best_child.parent_action)
        return best_child, normalized_distribution
    
    def get_distribution(self, root_node): #Include the fact that we have negative values
        distribution = []
        for child in root_node.children:
            distribution.append(child.visits)
        normalized_distribution = [float(i)/sum(distribution) for i in distribution] #Kan hende dette blir feil mtp neg verdier
        return normalized_distribution
    
    def get_distribution2(self, root_node): #Normalized visit counts. Noe muffins
        distribution = []
        all_actions = self.game.get_legal_actions()

        for action in all_actions:
            action_to_child = False
            for child in root_node.children:
                if child.parent_action == action:
                    distribution.append(child.visits)
                    action_to_child == True
                    break

            if not action_to_child:
                distribution.append(0)

        sum_distribution = sum(distribution)
        
        # Check if sum_distribution is not zero to avoid division by zero error
        if sum_distribution != 0:
            normalized_distribution = [float(i) / sum_distribution for i in distribution]
        else:
            # If sum_distribution is zero, return a uniform distribution
            num_actions = len(all_actions)
            normalized_distribution = [1 / num_actions] * num_actions
        return normalized_distribution
        
    def get_distribution3(self, root_node):
        print(len(root_node.children))
        print(len(self.game.get_legal_actions_with_0()))
              
        distribution = []
        for child in root_node.children:
            if child.parent_action in self.game.get_legal_actions():
                distribution.append(child.visits)
            else:
                distribution.append(0)
        
        non_zero_values = [x for x in distribution if x != 0]
        norm_non_zero_values = [(x - min(non_zero_values)) / (max(non_zero_values) - min(non_zero_values)) for x in non_zero_values]
        normalized_distribution = []
        for val in distribution:
            if val == 0:
                normalized_distribution.append(0)
            else:
                normalized_distribution.append(norm_non_zero_values)
        return normalized_distribution
    
    def rollout(self, game):
        current_rollout_state = game
        while not current_rollout_state.is_game_over():
            action = self.rollout_policy(current_rollout_state)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result() 
    
    def rollout_policy(self, current_rollout_state): # her skal ANET inn (ny klasse)
        possible_actions = current_rollout_state.get_legal_actions()
        valid_and_invalid_actions = current_rollout_state.get_legal_actions_with_0()
        self.anet.epsilon = self.anet.epsilon * 0.99 #want epsilon to get smaller
        if self.anet.get_epsilon() > np.random.rand():
            print("random choice")
            return random.choice(possible_actions)
        else:
            print("anet choice")
            return self.anet.predict(valid_and_invalid_actions, current_rollout_state) #returns best move based on distribution from Anet
            
    
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

        
        
    
    
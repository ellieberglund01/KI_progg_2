import numpy as np
import copy 
from math import sqrt, log,inf
import numpy as np

#PROBLEM 1: Når vi oppdaterer +1 for player 1 og -1 for player 2 wins, så får vi mange noder med 0 verdi -> Implementere q-funksjon?
#PROBLEM 2: Vi fortsetter backpropagation oppover i gammelt tre (ikke nldvendigvis noe problem)
#PROBLEM 3: Ved høye simuleringer så vil leaf policy gå ned til terminal state (ikke nødvendigvis et problem for hex)


class MCTS():

    def __init__(self, game_state, root_node, exploration_rate):
        self.game = game_state
        self.root_node = root_node
        self.exploration_rate = exploration_rate 

    def choose_action(self, root_node): #Policy, simulations, anet som input her også 
        """if root_node.parent != None:
            self.game.move(root_node.parent_action)"""
        
        print("Root node is now:", id(root_node))
        simulation_no = 4 #definere i init

        for i in range(1,simulation_no+1):
            print("Current simulation:", {i})
            print("-----------------")
            node = root_node
            game_copy = copy.deepcopy(self.game)
            '''
            if node.visits == 0:
                game_result = self.rollout(game_copy) 
                self.backpropagate(node, game_result, self.exploration_rate, i)
            
            else:
            '''
            leaf1, game_copy = self.tree_policy(node, game_copy, i) #explore best existing paths in the tree
            self.expand(leaf1, game_copy) #Adds children to leaf node
            expanded_leaf, game_copy = self.tree_policy(leaf1, game_copy, i) #Chooses beste action/leaf node from leaf1
            game_result = self.rollout(game_copy)  #Simulates game from expanded leaf
            path = self.backpropagate(expanded_leaf, root_node, game_result) #Backpropagates and returns the path from leaf to root
            for node in path:
                print("Path parent action:", node.parent_action, "Path visit count:", node.visits, "Path node value:", node.winning_count)

        #choosing the actual action in the game     
        child_values = []
        print("Possible actions and their values:")
        print("current player:", root_node.player)
        for child in root_node.children:
            print("Node value for root_action ",child.parent_action,"is",child.winning_count)
            if child.visits != 0:
                child_values.append(child.winning_count/child.visits) #Q-value
        if root_node.player == 1:
            best_child = root_node.children[np.argmax(child_values)]
        else:
            best_child = root_node.children[np.argmin(child_values)] 
        print(child_values)
        print("For player:", root_node.player, "best action is:", best_child.parent_action)
        return best_child
    

    def rollout(self, game): #Sende inn policy/ANET også
        current_rollout_state = game
        while not current_rollout_state.is_game_over():
            possible_actions = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_actions)
            print("Player:", game.player_turn, "takes random action",action)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result() 
    
    def rollout_policy(self, possible_moves): # her skal ANET inn (ny klasse)
        return possible_moves[np.random.randint(len(possible_moves))]
    
    def update_visit_count(self, node):
        while node != None:
            node.visits += 1
            node = node.parent

    def backpropagate(self, node, root, result):
        path = []
        self.update_visit_count(node)
        while node!= None: #Ender opp med å oppdatere visits + winning count for tidligere parent (er det dumt?)
            if result == 1:
                node.winning_count += 1
            else:
                node.winning_count -= 1 #Får mye w=0, som er dumt
            path.append(node)
            node = node.parent
        return path

    def backpropagate2(self,node,root,result): #denne oppdaterer q-funksjon. Da må vi endre treepolicy og calcUBC
        path = []
        self.update_visit_count(node)
        while node.parent_node!= None: 
            if result == 1:
                if node.parent_node.player == 1:
                    node.parent_node.qvalues[node.parent_action] +=1
            else:
                if node.parent_node.player == -1:
                    node.parent_node.qvalues[node.parent_action] -=1
            path.append(node)
            node = node.parent
        return path

    #Iterer gjennom treet basert på ubc scoren og finner actions som leder til leaf node i spillet 
            #burde kalkulere ubc på vei ned (ikke på vei opp)
    def tree_policy(self, node, game, sim):
        current_node = node
        actions = []
        print("TREE POLICY")
        print("number of children/actions;", len(current_node.children))
        while len(current_node.children) != 0:
            if current_node.player == 1:
                current_ubc = float('-inf')
                best_node = None
                for i in range (len(current_node.children)): #endre til branch?
                    if current_node.children[i].calcUBC(self.exploration_rate,sim) > current_ubc: #if ubc to child is higher than current ubc
                        best_node = current_node.children[i]
                        current_ubc = best_node.calcUBC(self.exploration_rate,sim)
                        print("current ubc in tree policy is", current_ubc, "for action", best_node.parent_action)
                current_node = best_node

            else:
                current_ubc = float('inf')
                best_node = None
                for i in range (len(current_node.children)):
                    if current_node.children[i].calcUBC(self.exploration_rate,sim) < current_ubc: #For player -1, we want to minimize ubc
                        best_node = current_node.children[i]
                        current_ubc =best_node.calcUBC(self.exploration_rate,sim)
                        print("current ubc in tree policy is", current_ubc, "for action", best_node.parent_action)
                current_node = best_node


            if current_node == None:
                raise Exception("No node found")
            actions.append(current_node.parent_action)
        
        for action in actions:
            game.move(action)
            print("In tree policy, action:",action, "is taken")
        print("----Tre policy done----")
        return current_node, game


    def create_child(self,parent, parent_action):
            if parent.player == 1: 
                child = Node(-1, parent, parent_action)
            else:
                child = Node(1, parent, parent_action)
            return child
    
    def expand(self,initial_node, game): #Expander alle action fra node og utvider treet 
        actions = game.get_legal_actions()
        for action in actions:
            child = self.create_child(initial_node, action)
            if child not in initial_node.children: 
                initial_node.children.append(child)
    
    def simluation_results(self, node):
        possible_actions = self.game.get_legal_actions()
        #lists of visists of each child in simulations 
        node_visits_list = [child.visits if child.parent_action == action else 0 for action in possible_actions for child in node.children]
        total_visists = sum(node_visits_list)
        distribution= [float(node_visits)/total_visists for node_visits in node_visits_list]
        return distribution
    
    
class Node():
    def __init__(self, player, parent=None, parent_action=None):
        self.player = player
        self.parent = parent
        self.parent_action = parent_action
        self.children=[]
        self.winning_count = 0 #blir feil å kalle det winning count, fordi blir både +1 og -1 (er mer node_value)
        self.visits = 0
        self.qvalues = {}

        if self.player == -1:
            self.ubc = float('inf')
        else:
            self.ubc = float('-inf')
    
    def is_terminal_node(self): 
        return self.state.is_game_over()

    def update(self, result, player, exploration_rate, sim): #Brukes ikke nå
        if result == player:
            if self.player == player:
                self.winning_count += 1
        '''
        if result == 1: #player 1 wins. + 1
            self.winning_count += 1
        else: #player -1 wins. +0
            self.winning_count -= 1
        #self.ubc = self.calcUBC(exploration_rate, sim)
        #If root.player = rollout.result + 1. In entire path
        '''
    def calcUBC(self,exploration_rate, sim): #Simulations er en del av utregning av ubc 
        w_i = self.winning_count #this nodes number of simluations that resultet in win 
        s_i = self.visits #this nodes total number of visists/simulations 
        c = exploration_rate

        if self.visits == 0:
            if self.player == 1:
                self.ubc = float("-inf")
            else:
                self.ubc = float("inf")
            return self.ubc
        
        if self.parent == None: #root node 
            self.ubc = w_i/s_i + c* sqrt(np.log(sim)/s_i)
        
        else:
            s_p = self.parent.visits #parent node's total number of visits/ simulations 

            if self.player == 1: 
                self.ubc = w_i/s_i + c* sqrt(np.log(s_p)/s_i)
            else:
                self.ubc = w_i/s_i - c* sqrt(np.log(s_p)/s_i)
        return self.ubc
    
    
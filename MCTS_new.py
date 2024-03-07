import numpy as np
import copy 
from math import sqrt, log,inf
import numpy as np



class MCTS():

    def __init__(self, game_state, root_node, exploration_rate):
        self.game = game_state
        self.root_node = root_node
        self.exploration_rate = exploration_rate 

    #Egentlig mer run simulation funcion
    def choose_action(self, root_node): #Policy, simulations, anet som input her også 
        """if root_node.parent != None:
            self.game.move(root_node.parent_action)"""

        simulation_no = 3 #definere i init

        for i in range(1,simulation_no-1):
            print("Current simulation:", {i})
            node = root_node
            game_copy = copy.deepcopy(self.game)
            leaf1, game_copy = self.tree_policy(node, game_copy)
            print(leaf1.visits)
            """
            if leaf1.visits == 0:
                print("leaf1 has 0 visits")
                game_result = self.rollout(game_copy) #policy er også input her 
                self.backpropagate(node, game_result, self.exploration_rate, i) 
            
            else:
            """
            expanded_node = self.expand(leaf1, game_copy) #Evt. behlde denne og ikke if setningen over 
            leaf2, game_copy = self.tree_policy(expanded_node, game_copy) #Velger beste action etter utvidelse av treet 
            game_result = self.rollout(game_copy)  #policy er også input her egentlig
            self.backpropagate(leaf2, game_result,self.exploration_rate, i) 

        #choosing the actual action 
        child_values = []
        for child in root_node.children:
            if child.visits != 0:
                child_values.append(child.winning_count/child.visits)
            print("child visits", child.visits)
        print("child values:", child_values)
        best_child = root_node.children[np.argmax(child_values)]
        print("best action",best_child.parent_action)
        
        return best_child


    def rollout(self, game): #Sende inn policy/ANET også
        current_rollout_state = game
        while not current_rollout_state.is_game_over():
            possible_actions = current_rollout_state.get_legal_actions()
            #print("Possible actions:", possible_actions)
            action = self.rollout_policy(possible_actions)
            #print("action in simulated game:", action)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result() 
    
    def rollout_policy(self, possible_moves): # her skal ANET inn (ny klasse)
        return possible_moves[np.random.randint(len(possible_moves))]
    
    def update_visit_count(self, node):
        while node != None:
            node.visits += 1
            node = node.parent

    def backpropagate(self,node, result,exploration_rate, sim):
        self.update_visit_count(node)
        while node!= None:
            node.update(node.player,result,exploration_rate, sim)
            node = node.parent

    #Iterer gjennom treet basert på ubc scoren og finner actions som leder til leaf node i spillet 
    def tree_policy(self, node, game):
        current_node = node
        actions = []
        while len(current_node.children) != 0:
            if current_node.player == 1:
                current_ubc = float('-inf')
                best_node = None
                

                for i in range (len(current_node.children)):
                    if current_node.children[i].ubc > current_ubc:
                        best_node = current_node.children[i]
                        current_ubc =best_node.ubc
                current_node = best_node

            else:
                current_ubc = float('inf')
                best_node = None
                for i in range (len(current_node.children)):
                    if current_node.children[i].ubc < current_ubc:
                        best_node = current_node.children[i]
                        current_ubc =best_node.ubc
                current_node = best_node

            if current_node == None:
                raise Exception("No node found")
            actions.append(current_node.parent_action)

        for action in actions:
            game.move(action)
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
        return initial_node
    
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
        self.winning_count = 0
        self.visits = 0

        if self.player == -1:
            self.ubc = float('inf')
        else:
            self.ubc = float('-inf')
    
    def is_terminal_node(self): 
        return self.state.is_game_over()

    def update(self, result, player, exploration_rate, sim): 
        if result == 1: #player 1 wins
            self.winning_count += 1
        else: #player -1 wins
            self.winning_count -= 1
        self.ubc = self.calcUBC(exploration_rate, sim)

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
                print(s_p)
                print("s_i", s_i)
                self.ubc = w_i/s_i - c* sqrt(np.log(s_p)/s_i)

        return self.ubc

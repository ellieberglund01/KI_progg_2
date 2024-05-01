import matplotlib.pyplot as plt
import networkx as nx

class DisplayGame():
    def __init__(self, hexGame):
        self.game = hexGame
        self.size =hexGame.board_size
        self.graph = nx.Graph()
        self.frame_delay = 0.5
        self.legal_positions = self.get_legal_positions()
        self.edges = set([(0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0)])

        for i in range(self.size):
            for j in range(self.size):
              self.graph.add_node((i, j))
        
        for x, y in self.legal_positions:
            for row_off, col_off in self.edges:
                neighbor_node = (x + row_off, y + col_off)
                if neighbor_node in self.legal_positions:
                    self.graph.add_edge((x, y), neighbor_node)

    def get_legal_positions(self):
        legal_positions = []
        for row in range(self.size):
            for col in range(self.size):
                legal_positions.append((row, col))
        return legal_positions
    
    def get_filled_nodes(self):
        filled1 = []
        filled2 = []
        for position in self.get_legal_positions():
            row = position[0]
            col = position[1]
            if self.game.board[row][col] == (1,0):
                filled1.append(position)
            if self.game.board[row][col] == (0,1):
                filled2.append(position)
        return filled1, filled2
    
    def get_empty_nodes(self):
        empty_positions = []
        for position in self.get_legal_positions():
            row = position[0]
            col = position[1]
            if self.game.board[row][col] == (0,0):
                empty_positions.append(position)
        return empty_positions

    def draw_board(self, winner, player1, player2):
        filled1, filled2 = self.get_filled_nodes()
        empty_nodes = self.get_empty_nodes()
        legal_positions = self.get_legal_positions()
        positions = {}
        winning_nodes = []

        # Position nodes to shape a Diamond
        for node in legal_positions:
            positions[node] = (node[0] - node[1], 2 * self.size - node[1] - node[0])

        plt.axis('off')
        # Drawing in the nodes that have not been taken
        nx.draw_networkx_nodes(self.graph, pos=positions, nodelist=empty_nodes, node_color='white')
        nx.draw_networkx_edges(self.graph, pos=positions, alpha=0.5, width=1, edge_color='black')
    

        if winner == 1:
            nx.draw_networkx_nodes(self.graph, pos=positions, nodelist=filled1, node_color='red', label=player1)
            nx.draw_networkx_nodes(self.graph, pos=positions, alpha=0.5, nodelist=filled2, node_color='grey', label=player2)
            winning_nodes = self.get_filled_nodes()[0]  # Player 1's filled nodes
            edge_color = 'red'
       
        elif winner == 2:
            nx.draw_networkx_nodes(self.graph, pos=positions, nodelist=filled1, node_color='grey', label=player1)
            nx.draw_networkx_nodes(self.graph, pos=positions, alpha=0.5, nodelist=filled2, node_color='blue', label=player2)
            winning_nodes = self.get_filled_nodes()[1]  # Player 2's filled nodes
            edge_color = 'blue'
        
        else:
            nx.draw_networkx_nodes(self.graph, pos=positions, nodelist=filled1, node_color='red', label=player1)
            nx.draw_networkx_nodes(self.graph, pos=positions, alpha=0.5, nodelist=filled2, node_color='blue', label=player2)
            winning_nodes = []
            edge_color = 'black'

        #Coloring edges connecting winning nodes
        #Error: Alle edges mellom alle noder til winner player blir farget, ikke bare winning path 
        for edge in self.graph.edges():
            if edge[0] in winning_nodes and edge[1] in winning_nodes:
                nx.draw_networkx_edges(self.graph, pos=positions, edgelist=[edge], edge_color=edge_color)
        
        plt.legend(prop={'size': 12})
        plt.draw()
        plt.pause(self.frame_delay)
        plt.show()


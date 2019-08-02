import numpy as np
from collections import defaultdict

class MoveGame(object):
    def __init__(self, x_coordinate, y_coordinate, value):
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.value = value
    def __repr__(self):
        return "x:{0} y: {1} v: {2}".format(self.x_coordinate, self.y_coordinate, self.value)
    

class MonteCarloTreeSearchNode(object):

    def __init__(self, state, parent=None):
        """
        Parameters
        ----------
        state : mctspy.games.common.TwoPlayersAbstractGameState
        parent : MonteCarloTreeSearchNode
        """
        self.state = state
        self.parent = parent
        self.children = []
    def is_fully_expanded(self):
            return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):        
        return possible_moves[np.random.randint(len(possible_moves))]
    
     
class TwoPlayersGameMonteCarloTreeSearchNode(MonteCarloTreeSearchNode):

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self._number_of_visits = 0.
        self._results = defaultdict(int)
        self._untried_actions = None

   
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    
    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.move(action)
        child_node = TwoPlayersGameMonteCarloTreeSearchNode(
            next_state, parent=self
        )
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)
    

class GameState(object):
    corn = 1
    beans = -1 
    cscore = 0
    score, bscore = 0
    def __init__(self, state):
        self.state = state
        self.board_size = state.shape[0]
    
    def is_game_over(self):
        return self.game_result is not None
    
    def check_score(self, x, y):
        self.x = x
        self.y = y
        
        for i in range(x):
            for j in range(y):
                if self.new_board[i][j] == self.beans:
                    self.cscore +=1
                
                
        return self.cscore
                    
        
    
    def game_result(self, new_board):
        # check if game is over
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.new_board[i][j] == self.corn:
                    if i-8< 0:
                        i = 0
                    if j-8< 0:
                        j =0
                    self.cscore = 10 + self.checkscore(i, j)
                    
                if self.new_board[i][j] == self.beans:
                    if i-8< 0:
                        i = 0
                    if j-8< 0:
                        j =0
                    self.bscore +=10
                
        return self.bscore+self.cscore
    
    def is_move_legal(self, move):
        if move.value!=0:
            return False
        else:
            return True
    def move(self, move):
        if not self.is_move_legal(move):
            raise ValueError('Move is not legal')
        self.new_board = np.copy(self.state)
        self.new_board[move.x_coordinate, move.y_coordinate] = move.value
        return GameState(self.new_board)
    
    def get_legal_actions(self):
        self.indices = np.where(self.board == 0)
        return [MoveGame(coords[0], coords[1]) for coords in list(zip(self.indices[0], self.indices[1]))]   
    
class MonteCarloTreeSearch(object):
    def __init__(self, node):
        self.root = node
    
    def rollout(self):
        current_rollout_state = self.state
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result
    
    def best_action(self, simulations_number):
        for _ in range(0, simulations_number):            
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        # to select best child go for exploitation only
        return self.root.best_child(c_param=0)
    def _tree_policy(self):
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
    


state = np.zeros((5,5))
#print initial_state
initial_board_state = GameState(state)
root = TwoPlayersGameMonteCarloTreeSearchNode(initial_board_state)
mcts = MonteCarloTreeSearch(root)
best_node = mcts.best_action(10000)

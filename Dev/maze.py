import random
import numpy as np
#from .game import Game
from game import Game

actions = {0:'left', 1:'right', 2:'up', '3':'down'}

class Maze(Game):
    def __init__(self, grid_size=10):
        self.grid_size=10
    
    @property
    def name(self):
        return "Maze"
    @property
    def nb_actions(self):
        return 4
    
    def reset(self):
        pass
    
    def play(self, action):
        pass
    
    def get_score(self):
        pass
    
    def get_state(self):
        pass
    
    def is_over(self):
        pass

    def is_won(self):
        pass


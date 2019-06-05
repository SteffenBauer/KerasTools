import random
import numpy as np

class Game(object):
    def __init__(self): self.reset()

    @property
    def name(self): return "Game"
    @property
    def nb_actions(self): return 0

    def reset(self): pass
    def play(self, action): pass
    def get_state(self): return None
    def get_score(self): return 0
    def is_over(self): return False
    def is_won(self): return False
    def get_frame(self): return self.get_state()
    def draw(self): return self.get_state()
    def get_possible_actions(self): return range(self.nb_actions)

actions = {0:'glide', 1:'flap'}

class Flappy(Game):

    def __init__(self, width=10, height=12, max_turn=None):
        self.width = width
        self.height = height
        self.max_turn = max_turn
        self.reset()

    @property
    def name(self):
        return "Flappy"
    @property
    def nb_actions(self):
        return 2
        
    def reset(self):
        self.lost = False
        self.turn = 0
        self.grid = [[0 for _ in range(self.width)] for _ in range(self.height)]

    def play(self, action):
        return self.get_score()

    def get_score(self):
        return 0

    def get_state(self):
        canvas = np.zeros((self.height,self.width,3))
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] == 1:
                    canvas[i,j,:] = (1,1,0)
        return canvas
    
    def is_over(self):
        if self.max_turn is not None and self.turn >= self.max_turn:
            return True
        else:
            return self.lost

    def is_won(self):
        return self.is_over() and not self.lost


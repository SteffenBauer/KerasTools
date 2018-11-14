import random
import numpy as np
from .game import Game

actions = {0:'left', 1:'idle', 2:'right'}

class Catch(Game):

    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        n = random.randrange(0, self.grid_size-1)
        m = random.randrange(1, self.grid_size-2)
        self.state = [0, n, m]

    @property
    def name(self):
        return "Catch"

    @property
    def nb_actions(self):
        return 3

    def play(self, action):
        if self.is_over() or action not in range(self.nb_actions):
            return
        fy, fx, basket = self.state
        if action == 0:
            new_basket = max(1, basket-1)
        elif action == 2:
            new_basket = min(self.grid_size-1, basket+1)
        else:
            new_basket = basket
        self.state = [fy+1, fx, new_basket]

    def get_state(self):
        im_size = (self.grid_size,) * 2
        fy, fx, basket = self.state
        canvas = np.zeros(im_size)
        canvas[-1, basket-1:basket + 2] = 0.7
        if self.is_won():
            canvas[fy, fx] = 0.5
        elif self.is_over():
            canvas[fy, fx] = 0.9
        else:
            canvas[fy, fx] = 0.7

        return canvas

    def get_score(self):
        if self.is_won():
            return 1
        elif self.is_over():
            return -1
        else:
            return 0

    def is_over(self):
        return self.state[0] == self.grid_size-1

    def is_won(self):
        fy, fx, basket = self.state
        return self.is_over() and abs(fx - basket) <= 1

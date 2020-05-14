import random
import numpy as np
from .game import Game

class Fruit(Game):
    def __init__(self, grid_size=12, max_turn = 24, with_border=True, with_poison=False, with_penalty=True, fixed=False):
        self.grid_size = grid_size
        self.with_border = with_border
        self.with_poison = with_poison
        self.fixed = fixed
        self.penalty = -0.1/float(max_turn) if with_penalty else 0.0
        self.max_turn = max_turn
        self.reset()

    @property
    def name(self):       return "Fruit"
    @property
    def nb_actions(self): return 4
    @property
    def actions(self):    return {0: 'left', 1: 'right', 2: 'up', 3: 'down'}
    
    def _random_coords(self):
        x = random.randrange(0, self.grid_size-1)
        y = random.randrange(0, self.grid_size-1)
        return x,y

    def reset(self):
        xa,ya = (0,0) if self.fixed else self._random_coords()
        while True:
            xf,yf = (self.grid_size-1, self.grid_size-1) if self.fixed else self._random_coords()
            if (xf!=xa) or (xf!=ya): break
        self.xa, self.ya = xa,ya
        self.xf, self.yf = xf,yf

        if self.with_poison:
            while True:
                xp,yp = (int(self.grid_size/2), int(self.grid_size/2)) if self.fixed else self._random_coords()
                if ((xp!=xa) or (yp!=ya)) and ((xp!=xf) or (yp!=yf)): break
            self.xp, self.yp = xp,yp

        self.eaten = False
        self.bumped = False
        self.poisoned = False
        self.starved = False
        self.turn = 0

    def play(self, action):
        if self.is_over() or (action not in range(self.nb_actions)):
            return

        if action == 0: self.ya -= 1
        if action == 1: self.ya += 1
        if action == 2: self.xa -= 1
        if action == 3: self.xa += 1

        self.check_fruit()
        if not self.is_over():                      self.check_border()
        if self.with_poison and not self.is_over(): self.check_poison()
        if not self.is_over():                      self.check_starved()
        
        return (self.get_frame(), self.get_score(), self.is_over())

    def check_fruit(self):
        if (self.xa == self.xf) and (self.ya == self.yf):
            self.eaten = True

    def check_border(self):
        if self.ya < 0:
            self.ya = 0
            if (self.with_border): self.bumped = True
        if self.ya >= self.grid_size:
            self.ya = self.grid_size-1
            if (self.with_border): self.bumped = True
        if self.xa < 0:
            self.xa = 0
            if (self.with_border): self.bumped = True
        if self.xa >= self.grid_size:
            self.xa = self.grid_size-1
            if (self.with_border): self.bumped = True

    def check_poison(self):
        if self.with_poison and (self.xa == self.xp) and (self.ya == self.yp):
            self.poisoned = True
   
    def check_starved(self):
        self.turn += 1
        if (self.max_turn is not None) and (self.turn > self.max_turn):
            self.starved = True

    def get_state(self):
        canvas = np.zeros((self.grid_size,self.grid_size,3))
        canvas[self.xa, self.ya, :] = (0.5,0.5,0.5) # Grey mouse
        #canvas[self.xa, self.ya, :] = (0,0,1) # Blue mouse
        canvas[self.xf, self.yf, :] = (1,1,0) # Yellow fruit
        if self.with_poison:
            canvas[self.xp, self.yp, :] = (0,1,1) # Cyan poison
        if self.starved or self.bumped or self.poisoned:
            canvas[self.xa, self.ya, :] = (1,0,0) # Red mouse if lost
        if self.eaten:
            canvas[self.xa, self.ya, :] = (0,1,0) # Green mouse if won
        return canvas

    def get_score(self):
        if self.starved:
            score = 0
        elif self.is_lost():
            score = -1
        elif self.eaten:
            score = 1
        else:
            score = self.penalty
        return score

    def is_lost(self):
        return self.starved or (self.with_border and self.bumped) or (self.with_poison and self.poisoned)

    def is_over(self):
        return self.eaten or self.is_lost()

    def is_won(self):
        return self.is_over() and not self.is_lost()


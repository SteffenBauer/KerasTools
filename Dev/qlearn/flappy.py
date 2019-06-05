import random
import numpy as np
from game import Game

class Flappy(Game):

    def __init__(self, width=288, height=512, difficulty=0, max_turn=None):
        self.width = width
        self.height = height
        self.difficulty = difficulty
        self.max_turn = max_turn
        self.pipegap = {0: 160, 1: 120, 2: 100 }[difficulty]
        self.pipewidth = 52
        self.birdsize = (34,24)
        self.boardsize = (288,512)
        self.bordersize = 32

        self.gravAcc = 1        # Downwards gravity acceleration
        self.flapAcc = -9       # Upwards flapping acceleration
        self.VelLimits = (10, -8) # Bird up/down velocity constraints
        self.pipeVel = -4       # Pipe leftwards velocity
        self.birdVel = 0        # Bird up/down velocity (variable)
        
        self.reset()

    @property
    def name(self):       return "Flappy"
    @property
    def nb_actions(self): return 2
    @property
    def actions(self):    return {0:'glide', 1:'flap'}

    def reset(self):
        self.lost = False
        self.turn = 0
        self.grid = [[0 for _ in range(self.width)] for _ in range(self.height)]

        self.birdPos = (self.boardsize[1]-self.birdsize[1])/2
    
    def _get_newpipe(self):
        pass

    def _check_crash():
        if self.birdPos < self.bordersize: return True
        if self.birdPos > (self.boardsize[1]-self.birdsize[1]-self.bordersize): return True
        return False
        
    def play(self, action):
        if not self.is_over():
            if action == 1: self.birdVel += self.flapAcc
            self.birdVel += self.gravAcc
            self.birdVel = min(self.birdVel, self.VelLimits[0])
            self.birdVel = max(self.birdVel, self.VelLimits[1])
        
            self.birdPos += self.birdVel
        
            self.lost = self._check_crash()
            
            self.birdPos = max(self.birdPos, self.bordersize)
            self.birdPos = min(self.birdPos, self.boardsize[1]-self.birdsize[1]-self.bordersize)
            
        return #(self.get_frame(), self.get_score(), self.is_over())

    def get_score(self):
        if self.is_won(): return 1
        if self.lost: return -1
        return 0

    def get_state(self):
        canvas = np.zeros((self.height,self.width,3))
    
        # Bird
        canvas[self.birdPos:self.birdPos+self.birdsize[1],
               self.width/2:self.birdsize[0]+self.width/2,:] = (1,1,0) if not self.lost else (1,0,0)

        # Top/Bottom borders
        canvas[:self.bordersize,:,:] = (0,1,0)
        canvas[self.boardsize[1]-self.bordersize:,:,:] = (0,1,0)

        return canvas
    
    def is_over(self):
        if self.max_turn is not None and self.turn >= self.max_turn:
            return True
        else:
            return self.lost

    def is_won(self):
        return self.is_over() and not self.lost


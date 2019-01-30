import random
import numpy as np
from .game import Game

actions = {0:'left', 1:'right', 2:'up', '3':'down'}

class Maze(Game):
    def __init__(self):
        self.width = 5
        self.height = 5
        self.reset()

    @property
    def name(self):
        return "Maze"
    @property
    def nb_actions(self):
        return 4

    def reset(self):
        self.grid = [[0 for i in range(self.width)] for j in range(self.height)]
        barriers = ((0,2),(2,1),(2,2),(2,3),(2,4),(4,0),(4,3))
        for bx,by in barriers:
            self.grid[bx][by] = 1
        self.grid[0][0] = 2
        self.grid[4][4] = 3
        self.x, self.y = 0,4
        self.bumped = False
        self.fallen = False
        self.won = False

    def play(self, action):
        if self.fallen or self.won:
            return
        dx,dy = ((-1,0),(1,0),(0,-1),(0,1))[action]
        nx,ny = self.x+dx, self.y+dy
        self.bumped = False
        self.fallen = False
        self.won = False
        if (nx < 0) or (nx>=self.width) or (ny < 0) or (ny >= self.height):
            self.bumped = True
        elif self.grid[nx][ny] == 1:
            self.bumped = True
        elif self.grid[nx][ny] == 2:
            self.fallen = True
        elif self.grid[nx][ny] == 3:
            self.won = True
        if not self.bumped:
            self.x, self.y = nx,ny

    def get_score(self):
        if self.fallen: return -1
        if self.won: return 1
        if self.bumped: return -0.5
        return -0.1

    def get_state(self):
        canvas = np.asarray(self.grid).astype('float32')
        canvas[self.x][self.y] = 4.0
        return canvas/4.0

    def is_over(self):
        return self.fallen or self.won

    def is_won(self):
        return self.won


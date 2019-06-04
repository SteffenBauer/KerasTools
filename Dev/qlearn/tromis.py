import random
import numpy as np
from game import Game

class Tromis(Game):

    def __init__(self, width=10, height=12, max_turn=None):
        self.width = width
        self.height = height
        self.max_turn = max_turn
        self.reset()

    @property
    def name(self):       return "Tromis"
    @property
    def nb_actions(self): return 5
    @property
    def actions(self):    return {0:'move-left', 1:'move-right', 2:'skip', '3':'rotate-left', '4':'rotate-right'}
    
    def reset(self):
        self.lost = False
        self.turn = 0
        self.removed_rows = 0
        self.grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self._new_piece()
        self.drop_counter = 2

    trominos = ( ( ((0,0), (0,-1), (0,1)),
                   ((0,0), (-1,0), (1,0)),
                   ((0,0), (0,-1), (0,1)),
                   ((0,0), (-1,0), (1,0))
                 ),
                 ( ((0,0), (0,1), (1,0)),
                   ((0,0), (1,0), (1,1)),
                   ((0,1), (1,0), (1,1)),
                   ((0,0), (0,1), (1,1))
                 ) )
    
    def _new_piece(self):
        self.p_type = random.randrange(2)
        self.p_orient = random.randrange(4)
        start_range = 0-min(t[1] for t in self.trominos[self.p_type][self.p_orient])
        end_range = self.width-max(t[1] for t in self.trominos[self.p_type][self.p_orient])
        self.p_row    = 1 if self.p_type == 0 else 0
        self.p_column = random.randrange(start_range, end_range)
        return

    def _valid_position(self, row, column, orient):
        for r,c in self.trominos[self.p_type][orient]:
            if (row+r < 0) or (row+r >= self.height):
                return False
            if (column+c < 0) or (column+c >= self.width):
                return False
            if self.grid[row+r][column+c] != 0:
                return False
        return True

    def _remove_completed_rows(self):
        removed_rows = 0
        while True:
            for i in range(len(self.grid)):
                if all(self.grid[i]): break
            else:
                break
            self.grid = [[0 for _ in range(self.width)]] + self.grid[0:i] + self.grid[i+1:]
            removed_rows += 1
        return removed_rows
        
    def play(self, action):
        if self.is_over() or (action not in range(self.nb_actions)):
            return

        if not self._valid_position(self.p_row, self.p_column, self.p_orient):
            self.lost = True
            return (self.get_frame(), self.get_score(), self.is_over())
            
        actions = [lambda r,c,o: (r,c-1,o),
                   lambda r,c,o: (r,c+1,o),
                   lambda r,c,o: (r,c,o),
                   lambda r,c,o: (r,c,(o+1)%4),
                   lambda r,c,o: (r,c,(o-1)%4)]
        nrow,ncol,nori = actions[action](self.p_row, self.p_column, self.p_orient)

        if self._valid_position(nrow,ncol,nori):
            self.p_column, self.p_row, self.p_orient = ncol, nrow, nori

        if not self._valid_position(self.p_row+1, self.p_column, self.p_orient):
            for r,c in self.trominos[self.p_type][self.p_orient]:
                self.grid[self.p_row+r][self.p_column+c] = 1
            self.removed_rows = self._remove_completed_rows()
            self._new_piece()
        else:
            self.p_row += 1
            self.removed_rows = 0
        
        self.turn += 1

        return (self.get_frame(), self.get_score(), self.is_over())

    def get_score(self):
        if self.lost:
            return -1
        elif self.is_won():
            return 0
        return 1 if self.removed_rows > 0 else 0

    def get_state(self):
        canvas = np.zeros((self.height,self.width,3))
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] == 1:
                    canvas[i,j,:] = (1,1,0)
        if self.lost: color = (1,0,0)
        elif self.p_type == 0: color = (0,0,1)
        else: color = (0,1,0)
        for r,c in self.trominos[self.p_type][self.p_orient]:
            canvas[self.p_row+r,self.p_column+c,:] = color
        return canvas
    
    def is_over(self):
        if self.max_turn is not None and self.turn >= self.max_turn:
            return True
        else:
            return self.lost

    def is_won(self):
        return self.is_over() and not self.lost


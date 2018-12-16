import random
import numpy as np
#from .game import Game
from game import Game

actions = {0:'move-left', 1:'move-right', 2:'skip', '3':'rotate-left', '4':'rotate-right'}

class Tromis(Game):

    def __init__(self, width=10, height=12, max_turn=100):
        self.width = width
        self.height = height
        self.max_turn = max_turn
        self.reset()
        print self.piece

    @property
    def name(self):
        return "Tromis"
    @property
    def nb_actions(self):
        return 5
        
    def reset(self):
        self.grid = [[0 for i in range(self.width)] for j in range(self.height)]
        self.piece = self._new_piece()
    
    #                #
    # 0/0  ###  0/1  #
    #                #
    #
    # 1/0  ##  1/1 #  1/2  #  1/3 ##
    #      #       ##     ##       #
    #
    # Ranges
    #        0   |   1   |  2  |  3  
    #  0 | 1/W-1 | 0/W   |
    #  1 | 0/W-1 | 0/W-1 | 1/W | 1/W
    
    ranges_hori = (((-1,1),(0,0),(-1,1),(0,0)),
                   ((0,1),(0,1),(-1,0),(-1,0)))
    ranges_vert = (((0,0),(-1,1),(0,0),(-1,1)),
                   ((0,1),(-1,0),(-1,0),(0,1)))
    
    def _new_piece(self):
        p_type = random.randrange(2)
        p_orient = random.randrange(4)
        start_range = 0-self.ranges_hori[p_type][p_orient][0]
        end_range = self.width-self.ranges_hori[p_type][p_orient][1]
        p_row = 1
        p_column = random.randrange(start_range, end_range)
        return [p_type, p_orient, p_row, p_column]

    def play(self, action):
        pass
    
    def get_score(self):
        pass
    
    def get_state(self):
        canvas = np.asarray(self.grid)
        tro_type, orient, row, column = self.piece
        color = 1 if tro_type == 0 else 2
        canvas[row][column] = color
        if tro_type == 0 and (orient == 0 or orient == 2):
            canvas[row][column-1] = color
            canvas[row][column+1] = color
        if tro_type == 0 and (orient == 1 or orient == 3):
            canvas[row-1][column] = color
            canvas[row+1][column] = color
        if tro_type == 1 and orient == 0:
            canvas[row][column+1] = color
            canvas[row+1][column] = color
        if tro_type == 1 and orient == 1:
            canvas[row-1][column] = color
            canvas[row][column+1] = color
        if tro_type == 1 and orient == 2:
            canvas[row-1][column] = color
            canvas[row][column-1] = color
        if tro_type == 1 and orient == 3:
            canvas[row][column-1] = color
            canvas[row+1][column] = color
        return canvas/2.0
    
    def is_over(self):
        pass

    def is_won(self):
        pass

    

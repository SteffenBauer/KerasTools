import random
import numpy as np
#from .game import Game
from game import Game

actions = {0:'move-left', 1:'move-right', 2:'skip', '3':'rotate-left', '4':'rotate-right'}

class Tromis(Game):

    def __init__(self, width=10, height=12, max_turn=None):
        self.width = width
        self.height = height
        self.max_turn = max_turn
        self.reset()

    @property
    def name(self):
        return "Tromis"
    @property
    def nb_actions(self):
        return 5
        
    def reset(self):
        self.grid = [[0 for i in range(self.width)] for j in range(self.height)]
        self._new_piece()
        #self.drop_counter = 2
    
    #                #                 #
    # 0/0  ###  0/1  #  0/2  ###  0/3  #
    #                #                 #
    #
    # 1/0  ##  1/1 #  1/2  #  1/3 ##
    #      #       ##     ##       #
    #
    
    trominos = ( 
                 ( ((0,0), (0,-1), (0,1)),
                   ((0,0), (-1,0), (1,0)),
                   ((0,0), (0,-1), (0,1)),
                   ((0,0), (-1,0), (1,0))
                 ),
                 ( ((0,0), (0,1), (1,0)),
                   ((0,0), (1,0), (1,1)),
                   ((0,0), (1,-1), (1,0)),
                   ((0,0), (0,-1), (1,0))
                 )
               )
    
    def _new_piece(self):
        self.p_type = random.randrange(2)
        self.p_orient = random.randrange(4)
        start_range = 0-min(t[1] for t in self.trominos[self.p_type][self.p_orient])
        end_range = self.width-max(t[1] for t in self.trominos[self.p_type][self.p_orient])
        self.p_row, self.p_column = 1, random.randrange(start_range, end_range)
        return

    def play(self, action):
        if action == 0:
            pass
        elif action == 1:
            pass
        elif action == 2:
            pass
        elif action == 3:
            pass
        elif action == 4:
            pass
        # Drop the piece one row downwards
        


    def get_score(self):
        pass
    
    def get_state(self):
        canvas = np.asarray(self.grid).astype('float32')
        color = 0.5 if self.p_type == 0 else 1.0
        for r,c in self.trominos[self.p_type][self.p_orient]:
            canvas[self.p_row+r][self.p_column+c] = color
        return canvas
    
    def is_over(self):
        pass

    def is_won(self):
        pass


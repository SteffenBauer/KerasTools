class Game(object):
    
    def __init__(self): self.reset()

    @property
    def name(self): return "Game"
    @property
    def nb_actions(self): return 0
    @property
    def actions(self): return dict()

    def reset(self): pass
    def play(self, action): return (self.get_frame(), self.get_score(), self.is_over())
    def get_state(self): return None
    def get_score(self): return 0
    def is_over(self): return False
    def is_won(self): return False
    def get_frame(self): return self.get_state()
    def draw(self): return self.get_state()
    def get_possible_actions(self): return range(self.nb_actions)


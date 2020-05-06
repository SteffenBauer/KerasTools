class BatchPlay:
    def __init__(self, game, batch_size):
        self.game = game
        self.batch_size = batch_size
        self.batch = [self.game for _ in range(self.batch_size)]
        self.reset()
        
    def reset(self):
        for g in self.batch: g.reset()
    
    def rollover(self):
        for g in self.batch:
            if g.is_over(): g.reset()
    
    def get_frames(self):
        return [g.get_frame() for g in self.batch]
    
    def play(self, actions):
        return [g.play(a) for g,a in zip(self.batch, actions)]


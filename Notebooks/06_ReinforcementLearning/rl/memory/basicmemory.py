from .memory import Memory

import random
import numpy as np

class BasicMemory(Memory):
    def __init__(self, memory_size=65536):
        self.memory_size = memory_size
        self.memory = list()
        
    def remember(self, S, a, r, Snext, game_over):
        self.memory.append((S,a,r,Snext,game_over))
        if self.memory_size is not None and len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def get_batch(self, _model, batch_size):
        if len(self.memory) < batch_size:
            return False
        return random.sample(self.memory, batch_size)

    def reset(self):
        self.memory = list()


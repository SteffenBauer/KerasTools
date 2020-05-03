from .memory import Memory

import random
import numpy as np

class UniqMemory(Memory):
    def __init__(self, memory_size=65536):
        self.memory_size = memory_size
        self.memory = list()
        self.hashes = set()
        
    def remember(self, S, a, r, Snext, game_over):
        h = hash((S.tostring(), a, r, Snext.tostring(), game_over))
        if h not in self.hashes:
            self.memory.append([(S,a,r,Snext,game_over), h])
            self.hashes.add(h)
        if self.memory_size is not None and len(self.memory) > self.memory_size:
            self.hashes.remove(self.memory.pop(0)[1])

    def get_batch(self, _model, batch_size):
        if len(self.memory) < batch_size:
            return False
        return [e[0] for e in random.sample(self.memory, batch_size)]

    def reset(self):
        self.memory = list()
        self.hashes = set()


#!/usr/bin/env python

import random
import collections

class Memory:
    def __init__(self): pass
    def remember(self, S, a, r, S_prime, game_over): pass
    def get_batch(self, model, batch_size): pass
    def reset(self): pass

class BasicMemory(Memory):
    def __init__(self, memory_size=65536):
        self.memory_size = memory_size
        self.memory = collections.deque([], memory_size)

    def remember(self, S, a, r, Snext, game_over):
        entry = (S,a,r,Snext,game_over)
        #if entry not in self.memory:
        self.memory.append(entry)

    def get_batch(self, _model, batch_size):
        return random.sample(self.memory, min(len(self.memory), batch_size))

    def reset(self):
        self.memory = set()

if __name__ == '__main__':
    pass


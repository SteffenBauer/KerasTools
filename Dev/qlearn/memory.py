#!/usr/bin/env python

import random
import collections
import numpy as np

class Memory:
    """
    Abstract base class to built new replay buffers
    """
    def __init__(self): pass
    
    def remember(self, S, a, r, S_prime, game_over): 
        """
        Store one record in the replay buffer.
        
        # Arguments
            S:          Game state after the action. Numpy array in format `(nb_frames, height, width, channels)`
            a:          Integer. Played action.
            r:          Float. Received reward.
            S_prime:    Game state directly before the action. Same format as `S`.
            game_over:  Boolean. Game over in this state.

        """
        pass

    def get_batch(self, model, batch_size): 
        """
        Get a batch of replay records.
        
        # Arguments
            model:       Current network.
            batch_size:  Integer. Batch size.

        # Returns
            Batch of replay records. Format of one records see `remember`
        """
        pass

    def reset(self): 
        """Flush and empty the replay buffer"""
        pass

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

if __name__ == '__main__':
    pass


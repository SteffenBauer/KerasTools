#!/usr/bin/env python3

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

if __name__ == '__main__':
    pass


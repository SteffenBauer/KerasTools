#!/usr/bin/env python3

class Callback(object):
    """
    Abstract base class to built new callbacks
    """
    def __init__(self): pass

    def game_start(self, frame): 
        """
        Called when a new game is started during training

        # Arguments
            frame: Initial game state frame in format (height, width, channels)
        """
        pass

    def game_frame(self, frame): 
        """
        Called when the agent has played an action

        # Arguments
            frame: Game state frame after the action in format (height, width, channels)
        """
        pass

    def game_over(self): 
        """Called when the current game is over"""
        pass

    def epoch_end(self, *args): 
        """
        Called at the end of a training epoch

        # Arguments
            model:       DQN network
            name:        Name of the game
            epoch:       Current training epoch
            epsilon:     Current epsilon factor
            loss:        Training loss
            win_ratio    Percentage of won games in this epoch
            avg_score    Average game score in this epoch
            max_score    Highest game score in this epoch
            memory_fill  Records in the replay memory

        """
        pass

if __name__ == '__main__':
    pass


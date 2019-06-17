import time

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

class HistoryLog(Callback):
    def __init__(self, name):
        st = time.gmtime()
        self.timestamp = "{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}".format(st.tm_year, st.tm_mon, st.tm_mday, st.tm_hour, st.tm_min, st.tm_sec)
        self.filename = '{}-{}.log'.format(name, self.timestamp)
        with open(self.filename, 'w+') as fp:
            fp.write('Epoch, Epsilon,    Loss, Win Ratio, Avg Score, Max Score,   Memory\n')
    def epoch_end(self, *args):
        _model, name, epoch, epsilon, loss, win_ratio, avg_score, max_score, memory = args
        with open(self.filename, 'a') as fp:
            fp.write('{:> 5d}, {:>7.2f}, {:>7.4f}, {:>9.2%}, {:>9.2f}, {:>9.2f}, {:>8d}\n'.format(epoch, epsilon, loss, win_ratio, avg_score, max_score, memory))

class Checkpoint(Callback):
    def __init__(self, interval=1):
        self.interval = interval
    def epoch_end(self, *args):
        model, name, epoch, epsilon, loss, win_ratio, avg_score, max_score, memory = args
        if epoch % self.interval == 0:
            filename = '{}_{:03d}.h5'.format(name, epoch)
            model.save(filename)


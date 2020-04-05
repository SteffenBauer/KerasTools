from .callbacks import Callback

class Checkpoint(Callback):
    def __init__(self, interval=1):
        self.interval = interval
    def epoch_end(self, *args):
        model, name, epoch, epsilon, loss, win_ratio, avg_score, max_score, memory = args
        if epoch % self.interval == 0:
            filename = '{}_{:03d}.h5'.format(name, epoch)
            model.save(filename)


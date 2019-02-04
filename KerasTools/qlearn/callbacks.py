import time

class Callback(object):
    def game_start(self, frame): pass
    def game_frame(self, frame): pass
    def game_over(self): pass
    def epoch_end(self, *args): pass

class History(Callback):
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


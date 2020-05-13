from .callbacks import Callback

import time

class HistoryLog(Callback):
    def __init__(self, name, params=None):
        st = time.gmtime()
        self.timestamp = "{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}".format(st.tm_year, st.tm_mon, st.tm_mday, st.tm_hour, st.tm_min, st.tm_sec)
        self.filename = '{}-{}.log'.format(name, self.timestamp)
        with open(self.filename, 'w+') as fp:
            if params is not None:
                for k,v in sorted(params.items()):
                    fp.write("{:18s}: {}\n".format(k,v))
                fp.write("\n")
            fp.write('Epoch, Epsilon,    Loss, Win Ratio, Avg Score, Max Score,   Memory,      Timestamp\n')
    def epoch_end(self, *args):
        _model, name, epoch, epsilon, loss, win_ratio, avg_score, max_score, memory = args
        st = time.gmtime()
        timestamp = "{:04d}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}".format(st.tm_year, st.tm_mon, st.tm_mday, st.tm_hour, st.tm_min, st.tm_sec)
        with open(self.filename, 'a') as fp:
            fp.write('{:> 5d}, {:>7.2f}, {:>7.4f}, {:>9.2%}, {:>9.2f}, {:>9.2f}, {:>8d}, {}\n'.format(epoch, epsilon, loss, win_ratio, avg_score, max_score, memory, timestamp))


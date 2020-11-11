import numpy as np
import pandas as pd


class PerfData(object):
    def __init__(self, algo_name, epochs, save_eps=False):
        self.algo_name = algo_name
        self.tr_loss = np.zeros(epochs)
        self.te_loss = np.zeros_like(self.tr_loss)
        self.tr_acc = np.zeros_like(self.tr_loss)
        self.te_acc = np.zeros_like(self.tr_loss)
        self.etime = np.zeros_like(self.tr_loss)
        self.eps = np.zeros_like(self.tr_loss) if save_eps else None
        self.count = 0

    def add_perf_data(self, tr_loss, tr_acc, te_loss, te_acc, etime, eps=0.0):
        self.tr_loss.append(tr_loss)
        self.te_loss.append(te_loss)
        self.tr_acc.append(tr_acc)
        self.te_acc.append(te_acc)
        self.etime.append(etime)
        if self.eps is not None:
            self.eps.append(eps)

        self.count += 1

    def get_perf_data(self, index=-1):
        if self.count == 0:
            return None

        out = [self.tr_loss[index], self.tr_acc[index],
               self.te_loss[index], self.te_acc[index], self.etime[index]]
        
        if self.eps is not None:
            out.append(self.eps[index])

        return out

    def to_datafame(self):
        perf = {
            '{}_tr_loss'.format(self.algo_name): self.tr_loss,
            '{}_te_loss'.format(self.algo_name): self.te_loss,
            '{}_tr_acc'.format(self.algo_name): self.tr_acc,
            '{}_te_acc'.format(self.algo_name): self.te_acc,
            '{}_etime'.format(self.algo_name): self.etime,
        }

        if self.eps is not None:
            perf['{}_eps'.format(self.algo_name)] = self.eps

        return pd.DataFrame(perf)


class PerfLogger(object):
    def __init__(self, algo_names, epochs, save_eps=False):
        self._log_data = {algo_name: PerfData(alg_name, epochs, save_eps)
                          for algo_name in algo_names}

    def __getitem__(self, alg_name):
        return self._log_data[key]

    def __setitem__(self, alg_name, data):
        self._log_data[alg_name] = data

    def to_dataframe(self):
        dframes = [self._log_data[alg_name].to_datafame()
                   for alg_name in self.log_data]
        df = pd.concat(dframes, axis=1)
        df['epoch'] = range(1, len(dframe[0])+1)
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]

        return df

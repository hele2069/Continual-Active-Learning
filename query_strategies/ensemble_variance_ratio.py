import numpy as np
from .strategy import Strategy
import IPython

class Ensemble_VR(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, seeds):
        super(Ensemble_VR, self).__init__(X, Y, idxs_lb, net, handler, args, seeds)
    
    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        pred = self.predict_ensemble(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        vr = np.zeros(pred.shape[1])
        for i in range(pred.shape[1]):
            vr[i] = 1 - pred[:, i].bincount().max().item() / pred.shape[0]
        return idxs_unlabeled[vr.argsort()[-n:]]

import numpy as np
import torch
from .strategy import Strategy
import random
import IPython
import matplotlib.pyplot as plt
# plt.rcParams["figure.figsize"] = (20,3)

class NTKSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, X_val, Y_val):
        super(NTKSampling, self).__init__(X, Y, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)
        self.selected_idxs = []
        
    def query(self, n, pool_size=2000):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        idxs_unlabeled = idxs_unlabeled[random.sample(range(0, idxs_unlabeled.shape[0]), pool_size)]
        '''
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        log_probs = torch.log(probs)
        U = (probs*log_probs).sum(1)
        idxs_unlabeled = idxs_unlabeled[U.sort()[1][:pool_size]]
        '''
        score = self.estimate_ntk(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        # score = score + np.random.normal(0, 50, len(score))
        # selected_idxs = np.random.choice(idxs_unlabeled, n, p=score)
        selected_idxs = idxs_unlabeled[np.argsort(score)[-n:]]
        self.selected_idxs.append(selected_idxs)
        
        ''' 
        ------ plotting ntk distance ------
        if self.round > 0:
            score.sort()
            plt.plot(score[-n:], label='Round {}'.format(self.round))
            plt.legend()
            plt.savefig('score_dist.png')
        '''
        cb_vec = self.cb_vec[np.argsort(score)[-n:]].sum(0)
        return selected_idxs


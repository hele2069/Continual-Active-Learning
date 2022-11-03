import numpy as np
import torch
from .strategy import Strategy
import random

class BALDDropout(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args, n_drop=10, X_val=None, Y_val=None):
		super(BALDDropout, self).__init__(X, Y, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)
		self.n_drop = n_drop

	def query(self, n, pool_size=2000):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		idxs_unlabeled = idxs_unlabeled[random.sample(range(0, idxs_unlabeled.shape[0]), pool_size)]
		probs = self.predict_prob_dropout_split(self.X[idxs_unlabeled], self.Y[idxs_unlabeled], self.n_drop)
		pb = probs.mean(0)
		entropy1 = (-pb*torch.log(pb)).sum(1)
		entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
		U = entropy2 - entropy1
		return idxs_unlabeled[U.sort()[1][:n]]

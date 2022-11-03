import numpy as np
from .strategy import Strategy
import random

class MarginSampling(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args, X_val=None, Y_val=None):
		super(MarginSampling, self).__init__(X, Y, idxs_lb, net, handler, args,  X_val=X_val, Y_val=Y_val)

	def query(self, n, pool_size=2000):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		idxs_unlabeled = idxs_unlabeled[random.sample(range(0, idxs_unlabeled.shape[0]), pool_size)]

		probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		probs_sorted, idxs = probs.sort(descending=True)
		U = probs_sorted[:, 0] - probs_sorted[:,1]
		return idxs_unlabeled[U.sort()[1][:n]]

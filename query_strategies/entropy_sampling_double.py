import numpy as np
import torch
from .strategy import Strategy
import random
import IPython

class EntropySamplingDouble(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args, X_val, Y_val):
		super(EntropySamplingDouble, self).__init__(X, Y, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)

	def query(self, n, pool_size=2000):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		idxs_unlabeled = idxs_unlabeled[random.sample(range(0, idxs_unlabeled.shape[0]), pool_size)]
		probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		log_probs = torch.log(probs)
		U = (probs*log_probs).sum(1)
		return idxs_unlabeled[U.sort()[1][:n]], idxs_unlabeled[U.sort()[1][-n:]]

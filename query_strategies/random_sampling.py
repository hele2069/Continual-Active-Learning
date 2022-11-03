import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args, X_val, Y_val):
		super(RandomSampling, self).__init__(X, Y, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)

	def query(self, n, pool_size):
		return np.random.choice(np.where(self.idxs_lb==0)[0], n)

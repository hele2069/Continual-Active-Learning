import numpy as np
import torch
from .strategy import Strategy
import random
import IPython

# test on full train
# orthogonal similar to EWC (projectile, perpendicular, train on new task space)
# test results on entropy, margin, ntk, gnorm 
# point of GNorm: compare to NTK
class GNormSampling(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args, X_val, Y_val):
		super(GNormSampling, self).__init__(X, Y, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)

	def query(self, n, pool_size=2000):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		# idxs_unlabeled = idxs_unlabeled[random.sample(range(0, idxs_unlabeled.shape[0]), pool_size)]
		grad_emb = self.get_grad_embedding(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		
		U = np.linalg.norm(grad_emb, axis=1)
		return idxs_unlabeled[np.argsort(U)[-n:]]

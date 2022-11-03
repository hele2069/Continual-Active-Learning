import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy import stats
import random
import IPython

def init_centers(X, K):
	ind = np.argmax([np.linalg.norm(s, 2) for s in X])
	mu = [X[ind]]
	indsAll = [ind]
	centInds = [0.] * len(X)
	cent = 0
	while len(mu) < K:
		if len(mu) == 1:
			D2 = pairwise_distances(X, mu).ravel().astype(float)
		else:
			newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
			for i in range(len(X)):
				if D2[i] > newD[i]:
					centInds[i] = cent
					D2[i] = newD[i]
		D2 = D2.ravel().astype(float)
		Ddist = (D2 ** 2) / sum(D2 ** 2)
		customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
		ind = customDist.rvs(size=1)[0]
		while ind in indsAll:
			ind = customDist.rvs(size=1)[0]
		mu.append(X[ind])
		indsAll.append(ind)
		cent += 1
	return indsAll



class KGradSampling(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args, X_val, Y_val):
		super(KGradSampling, self).__init__(X, Y, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)

	def query(self, n, pool_size=2000):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		idxs_unlabeled = idxs_unlabeled[random.sample(range(0, idxs_unlabeled.shape[0]), pool_size)]
		grad_emb = self.get_grad_embedding(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		# grad_emb = self.get_embedding(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		chosen = init_centers(grad_emb.numpy(), n)
		return idxs_unlabeled[chosen]

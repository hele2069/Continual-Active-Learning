import numpy as np
import torch
from .strategy import Strategy
import random
import IPython
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy import stats
from sklearn.cluster import KMeans



def init_centers(X, K):
    ind = np.argmax(X.sum(1))
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0

    if len(X) <= K:
        return np.arange(len(X))

    while len(mu) < K and len(X) > 1:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
            # D2 = 5e-6 * pairwise_distances(X, mu).ravel().astype(float) \
            #             + pairwise_distances(X, mu, metric='cosine').ravel().astype(float)
            # IPython.embed()
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            # newD = 5e-6 * pairwise_distances(X, [mu[-1]]).ravel().astype(float) \
            #             + pairwise_distances(X, [mu[-1]], metric='cosine').ravel().astype(float)
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


def sample_points(X, K, NTK_score, n_clusters=20, threshold=10):
    X_one = X.copy()

    mean = np.mean(X)
    std = np.std(X)

    X_one[X_one < mean] = 0
    X_one[X_one >= mean] = 1
    IPython.embed()

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_one)

    x_cls = kmeans.labels_
    print(np.bincount(x_cls))
    chosen = []
    for i in range(n_clusters):
        x = X[x_cls == i]
        x_idx = np.arange(len(X))[x_cls == i]
        # cent = kmeans.cluster_centers_[i]
        # dist = pairwise_distances(x, [cent]).ravel().astype(float)
        # chosen.extend(x_idx[np.argsort(dist)[:10]])

        cent = init_centers(x, K=int(K / n_clusters))
        chosen.extend(x_idx[cent])

        # score = NTK_score[x_cls == i]
        # chosen.extend(x_idx[np.argsort(score)[-int(K / n_clusters):]])
    return chosen


class NTK_Clsuter(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, X_val, Y_val):
        super(NTK_Clsuter, self).__init__(X, Y, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)
        self.selected_idxs = []
        self.intop = []

    def query(self, n, pool_size=2000):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        score = self.estimate_ntk(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])

        # selected_idxs = np.argsort(score)[-pool_size:]
        selected_idxs = np.arange(len(score))

        # sorted_idxs = idxs_unlabeled[np.argsort(score)]
        # selected_idxs = random.sample(range(0, len(score)), pool_size)
        idxs_unlabeled = idxs_unlabeled[selected_idxs]

        # chosen = init_centers(self.cb_vec[selected_idxs], n)
        chosen = sample_points(self.cb_vec[selected_idxs], n, n_clusters=10, NTK_score=score[selected_idxs])

        '''
        cb_vec = self.cb_vec[selected_idxs].sum(0)        
        plt.bar(np.arange(len(cb_vec)), cb_vec, width = 10)
        plt.savefig('./results/{}/cb_dist_{}_{}.png'.format(self.data, self.round, 'NTK_cluster'), dpi=500)
        plt.clf()
        '''
        # idxs_unlabeled = idxs_unlabeled[random.sample(range(0, idxs_unlabeled.shape[0]), pool_size)]
        # score = self.estimate_ntk(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        # chosen = init_centers(self.cb_vec, n)

        # self.intop.append(np.mean([True if i in sorted_idxs[-5000:] else False for i in idxs_unlabeled[chosen]]))
        # IPython.embed()
        return idxs_unlabeled[chosen]


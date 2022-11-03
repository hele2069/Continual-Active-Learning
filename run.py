import numpy as np
from dataset import get_dataset, get_handler
from model import get_net
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
                                LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                                KMeansSampling, KCenterGreedy, BALDDropout, CoreSet, \
                                AdversarialBIM, AdversarialDeepFool, ActiveLearningByLearning, Ensemble_VR, WAAL, NTKSampling, \
                                EntropySamplingDouble, KGradSampling, GNormSampling, HistoryEntropySampling, NTK_Clsuter
import IPython

# parameters
SEED = 1
NUM_INIT_LB = 200 # 200, 2000
NUM_QUERY = 200 # 10, 100, 1000, 10000
# 100, 150, 200, 500
NUM_ROUND = 14 # NUM_QUERY * NUM_ROUND <= 30000

#DATA_NAME = 'MNIST'
# DATA_NAME = 'FashionMNIST'
# DATA_NAME = 'SVHN'
DATA_NAME = 'CIFAR10'

args_pool = {'MNIST':
                {'n_epoch': 80, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 128, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.05, 'momentum': 0.9}, 
                 'n_class': 10, 'data': 'MNIST'},
            'FashionMNIST': # 50
                {'n_epoch': 50, 'transform': transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}, 'n_class': 10},
            'SVHN':
                {'n_epoch': 100, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 256, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}, 'n_class': 10, 'data': 'SVHN'},
            'CIFAR10':
                {'n_epoch': 50, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'loader_tr_args':{'batch_size': 32, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.05, 'momentum': 0.3}, 'n_class': 10, 'data': 'CIFAR10'}
            }
args = args_pool[DATA_NAME]

def print_points(strategy, query_idxs):
    loader_tr = DataLoader(strategy.handler(strategy.X, strategy.Y, transform=args['transform'], total_size=None),
                            shuffle=False, **args['loader_te_args'])
    x_emb = []
    y_list = []
    for index, (x, y, idxs) in enumerate(loader_tr):
        x = x.to(strategy.device)
        output = strategy.clf(x)
        # y = output.max(1)[1].cpu()
        x_emb.append(strategy.clf.get_embedding(x).detach().cpu().numpy())
        y_list.append(y)
    # x_emb = np.stack(x_emb, 1)
    x_emb = np.concatenate(x_emb, 0)
    # x_emb = torch.cat(x_emb, 0)
    y_list = torch.cat(y_list, 0)
    x_r = TSNE(n_components=2).fit_transform(x_emb)
    for i in range(10):
        idxs = y_list == i
        plt.scatter(x_r[idxs, 0], x_r[idxs, 1], marker='x', label=i)
    plt.legend()
    plt.scatter(x_r[query_idxs, 0], x_r[query_idxs, 1], color='yellow')
    plt.savefig('./results/{}/{}_points_{}'.format(DATA_NAME, type(strategy).__name__, 
                        strategy.round))
    plt.clf()


# set seed
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.enabled = False

# load dataset
# X_tr, Y_tr, Y_tr_noisy, X_te, Y_te = get_dataset(DATA_NAME, noise_rate=0.3)
X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME)

X_val = X_tr[:500]
Y_val = Y_tr[:500]
X_tr = X_tr[500:]
Y_tr = Y_tr[500:]
# Y_tr_noisy = Y_tr_noisy[500:]

# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)
print('number of labeled pool: {}'.format(NUM_INIT_LB))
print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
print('number of testing pool: {}'.format(n_test))

# generate initial labeled pool
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_tmp = np.arange(n_pool)
np.random.shuffle(idxs_tmp)

for i in range(args['n_class']):
    idxs_lb[idxs_tmp[Y_tr == i][:int(NUM_INIT_LB / args['n_class'])]] = True

# load network
net = get_net(DATA_NAME)
handler = get_handler(DATA_NAME)

# strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)
# strategy = LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args)
# strategy = MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)

# strategy = EntropySampling(X_tr, Y_tr, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)
# strategy = LeastConfidenceDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=10)
# strategy = MarginSamplingDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=10)
# strategy = EntropySamplingDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=20, X_val=X_val, Y_val=Y_val)
# strategy = KMeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
# strategy = KCenterGreedy(X_tr, Y_tr, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)
# strategy = BALDDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=20, X_val=X_val, Y_val=Y_val)
# strategy = CoreSet(X_tr, Y_tr, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)
# strategy = AdversarialBIM(X_tr, Y_tr, idxs_lb, net, handler, args, eps=0.05)
# strategy = AdversarialDeepFool(X_tr, Y_tr, idxs_lb, net, handler, args, max_iter=50)
# albl_list = [MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args),
#              KMeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)]
# strategy = ActiveLearningByLearning(X_tr, Y_tr, idxs_lb, net, handler, args, strategy_list=albl_list, delta=0.1)
# strategy = Ensemble_VR(X_tr, Y_tr, idxs_lb, net, handler, args, seeds=[1, 12, 123, 1234, 12345])

strategy = NTKSampling(X_tr, Y_tr, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)
# strategy = EntropySamplingDouble(X_tr, Y_tr, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)
# strategy = KGradSampling(X_tr, Y_tr, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)
# strategy = GNormSampling(X_tr, Y_tr, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)
# strategy = HistoryEntropySampling(X_tr, Y_tr, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)
# strategy = NTK_Clsuter(X_tr, Y_tr, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)


# print info
print(DATA_NAME)
print('SEED {}'.format(SEED))
print(type(strategy).__name__)

# round 0 accuracy
strategy.new_selected = np.where(idxs_lb != 0)[0]
strategy.train()

P = strategy.predict(X_te, Y_te)
acc = np.zeros(NUM_ROUND+1)
acc[0] = 1.0 * (Y_te==P).sum().item() / len(Y_te)
print('Round 0\ntesting accuracy {}'.format(acc[0]))
class_entropy = []
class_dist = []

for rd in range(1, NUM_ROUND+1):
    print('Round {}'.format(rd))

    # strategy.estimate_fisher()
    # strategy.estimate_mas()
    strategy.estimate_z()

    q_idxs = strategy.query(NUM_QUERY, pool_size=5000)
    # q_idxs, p_idxs = strategy.query(NUM_QUERY, pool_size=10000)
    idxs_lb[q_idxs] = True
    
    # print_points(strategy, q_idxs)
    strategy.new_selected = q_idxs
    # strategy.new_selected_bottom = p_idxs
    
    cls_dist = strategy.Y[q_idxs].bincount().numpy() / len(q_idxs)
    class_entropy.append(np.sum(-cls_dist * np.log(cls_dist + 1e-5)))
    # print(class_entropy)
    class_dist.append(strategy.Y[q_idxs].bincount().numpy())

    # update
    strategy.update(idxs_lb)
    if rd <= 1000:
        strategy.train(use_history=False)
    else:
        strategy.train(use_history=True)

    # round accuracy
    P = strategy.predict(X_te, Y_te)
    acc[rd] = 1.0 * (Y_te==P).sum().item() / len(Y_te)
    print('testing accuracy: ', acc[rd])

# print results
print('SEED {}'.format(SEED))
print(type(strategy).__name__)
print(','.join([str(a) for a in acc]))
print('NUM Query: ', NUM_QUERY)
print('Class entropy: ', class_entropy)
IPython.embed()

import numpy as np
from dataset import get_dataset, get_handler, get_handler_WA
from model import get_net
from torchvision import transforms
import torch
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
                                LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                                KMeansSampling, KCenterGreedy, BALDDropout, CoreSet, \
                                AdversarialBIM, AdversarialDeepFool, ActiveLearningByLearning, Ensemble_VR, WAAL
import IPython

# parameters
SEED = 1

NUM_INIT_LB = 2000 # 200, 2000
NUM_QUERY = 2000 # 10, 100, 1000, 10000
# 100, 150, 200, 500
NUM_ROUND = 5 # NUM_QUERY * NUM_ROUND <= 30000
alpha = 2e-3

# DATA_NAME = 'MNIST'
# DATA_NAME = 'FashionMNIST'
# DATA_NAME = 'SVHN'
DATA_NAME = 'CIFAR10'

args_pool = {'MNIST':
                {'n_epoch': 50, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 128, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.001, 'momentum': 0.9}, 
                 'n_class': 10},
            'FashionMNIST':
                {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}, 'n_class': 10},
            'SVHN':
                {'n_epoch': 20, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}, 'n_class': 10},
            'CIFAR10':
                {'n_epoch': 50, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.005, 'momentum': 0.3}, 'n_class': 10}
            }
args = args_pool[DATA_NAME]

# set seed
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.enabled = False

# load dataset
# X_tr, Y_tr, Y_tr_noisy, X_te, Y_te = get_dataset(DATA_NAME, noise_rate=0.3)
X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME)

# IPython.embed()
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
net_fea, net_clf, net_dis = get_net(DATA_NAME+"_WAAL")
handler = get_handler(DATA_NAME)

train_handler = get_handler_WA(DATA_NAME)
test_handler = get_handler(DATA_NAME)
strategy = WAAL(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args, X_val=X_val, Y_val = Y_val)

# print info
print(DATA_NAME)
print('SEED {}'.format(SEED))
print(type(strategy).__name__)

# round 0 accuracy
strategy.train(alpha=alpha, total_epoch = args_pool[DATA_NAME]['n_epoch'])

P = strategy.predict(X_te, Y_te)
acc = np.zeros(NUM_ROUND+1)
acc[0] = 1.0 * (Y_te==P).sum().item() / len(Y_te)
print('Round 0\ntesting accuracy {}'.format(acc[0]))

for rd in range(1, NUM_ROUND+1):
    print('Round {}'.format(rd))

    q_idxs = strategy.query(NUM_QUERY)
    idxs_lb[q_idxs] = True

    # update
    strategy.update(idxs_lb)
    strategy.train(alpha=alpha, total_epoch = args_pool[DATA_NAME]['n_epoch'])

    # round accuracy
    P = strategy.predict(X_te, Y_te)
    acc[rd] = 1.0 * (Y_te==P).sum().item() / len(Y_te)
    print('testing accuracy: ', acc[rd])

# print results
print('SEED {}'.format(SEED))
print(type(strategy).__name__)
print(','.join([str(a) for a in acc]))
print('NUM Query: ', NUM_QUERY)
IPython.embed()

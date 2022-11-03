import numpy as np
from xmlrpc.client import boolean
import argparse
from dataset import get_dataset, get_handler
from model import get_net
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
                                LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                                KMeansSampling, KCenterGreedy, BALDDropout, CoreSet, \
                                AdversarialBIM, AdversarialDeepFool, ActiveLearningByLearning, Ensemble_VR, WAAL, NTKSampling, \
                                EntropySamplingDouble, KGradSampling, GNormSampling, HistoryEntropySampling, NTK_Clsuter
import IPython
import time 
from datetime import date

# sampling methods 
"""
choices = ["RandomSampling", 
            "LeastConfidence", 
            "MarginSampling", 
            "EntropySampling", 
            "LeastConfidenceDropout", 
            "MarginSamplingDropout", 
            "EntropySamplingDropout", 
            "KMeansSampling",
            "KCenterGreedy", 
            "BALDDropout", 
            "AdversarialBIM", 
            "AdversarialDeepFool",
            "NTKSampling"]
"""
# choices = ['NTKSampling', 'EntropySampling', 'MarginSampling']
# choices = ['NTKSampling']
choices = ['EntropySampling']

# parameters
SEED = 1
DATA_NAME = input('Enter the dataset to train (MNIST, FashionMNIST, SVHN, CIFAR10): ')
NUM_INIT_LB = input('Enter number of init labeled samples: ') # 50, 200, 2000
NUM_QUERY = input('Enter number of queries per round: ') # 10, 50, 100, 1000, 10000
NUM_ROUND = input('Enter number of rounds: ') # NUM_QUERY * NUM_ROUND <= 30000; can be 100, 150, 200, 500
TRAIN = input('Full or Continual: ') 
is_continual = False
if TRAIN == 'Continual': is_continual = True

# parse parameters
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=SEED, type=int)
parser.add_argument('--data', default=DATA_NAME, type=str)
parser.add_argument('--num_init', default=int(NUM_INIT_LB), type=int) # 50
parser.add_argument('--num_query', default=int(NUM_QUERY), type=int) # 50
parser.add_argument('--num_round', default=int(NUM_ROUND), type=int) # 10
parser.add_argument('--continual', default=is_continual, type=bool)
parser.add_argument('--online_ewc', default=False, type=bool)
parser.add_argument('--ewc_lambda', default=50, type=float)

args = parser.parse_args()

args_pool = {'MNIST':
                {'n_epoch': 50, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 128, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 128, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.05, 'momentum': 0.9}, 
                 'n_class': 10, 'data': 'MNIST'},
            'FashionMNIST': # 50
                {'n_epoch': 50, 'transform': transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}, 'n_class': 10, 'data': 'FashionMNIST'},
            'SVHN':
                {'n_epoch': 100, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 256, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}, 'n_class': 10, 'data': 'SVHN'},
            'CIFAR10':
                {'n_epoch': 40, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'loader_tr_args':{'batch_size': 32, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 32, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.001, 'momentum': 0.3}, 'n_class': 10, 'data': 'CIFAR10'}
            }
for arg in args_pool[args.data]:
    setattr(args, arg, args_pool[args.data][arg])

# print points
def print_points(strategy, query_idxs):
    loader_tr = DataLoader(strategy.handler(strategy.X, strategy.Y, transform=args.transform, 
                            total_size=None), shuffle=False, **args.loader_te_args)
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
    plt.savefig('./results/{}/{}_points_{}'.format(args.data, type(strategy).__name__, 
                        strategy.round))
    plt.clf()

# store output 
output = {'model': [],
          'round': [],
          'time': [],
          'accuracy': [],
          'size': [],
          'dataset':[],
          'strategy':[],
          'num_init_lb':[],
          'num_query':[],
          'num_round':[],
          'train':[]} 

# set seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.enabled = False

# load dataset
# X_tr, Y_tr, Y_tr_noisy, X_te, Y_te = get_dataset(args.data, noise_rate=0.3)
X_tr, Y_tr, X_te, Y_te = get_dataset(args.data)

X_val = X_tr[:500]
Y_val = Y_tr[:500]
X_tr = X_tr[500:]
Y_tr = Y_tr[500:]
# Y_tr_noisy = Y_tr_noisy[500:]

# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)
print('number of labeled pool: {}'.format(args.num_init))
print('number of unlabeled pool: {}'.format(n_pool - args.num_init))
print('number of testing pool: {}'.format(n_test))

# generate initial labeled pool
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_tmp = np.arange(n_pool)
np.random.shuffle(idxs_tmp)

# load network
net = get_net(args.data)
handler = get_handler(args.data)

for i in range(args.n_class):
    idxs_lb[idxs_tmp[Y_tr == i][:int(args.num_init / args.n_class)]] = True

for i in choices:

    # load strategy 
    STRATEGY_NAME = i
    #albl_list = [MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args),
    #            KMeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)]
    def get_strategy(name):
        if name == "RandomSampling":
            return RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)
        elif name == "LeastConfidence":
            return LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args)
        elif name == "MarginSampling":
            return MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)
        elif name == "EntropySampling":
            return EntropySampling(X_tr, Y_tr, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)
        elif name == "LeastConfidenceDropout":
            return LeastConfidenceDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=10)
        elif name == "MarginSamplingDropout":
            return MarginSamplingDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=10)
        elif name == "EntropySamplingDropout":
            return EntropySamplingDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=20, X_val=X_val, Y_val=Y_val)
        elif name == "KMeansSampling":
            return KMeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
        elif name == "KCenterGreedy":
            return KCenterGreedy(X_tr, Y_tr, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)
        elif name == "BALDDropout":
            return BALDDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=20, X_val=X_val, Y_val=Y_val)
        elif name == "AdversarialBIM":
            return AdversarialBIM(X_tr, Y_tr, idxs_lb, net, handler, args, eps=0.05)
        elif name == "AdversarialDeepFool":
            return AdversarialDeepFool(X_tr, Y_tr, idxs_lb, net, handler, args, max_iter=50)
        elif name == 'CoreSet':
            return CoreSet(X_tr, Y_tr, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)
        elif name == 'ActiveLearningByLearning':
            return ActiveLearningByLearning(X_tr, Y_tr, idxs_lb, net, handler, args, strategy_list=albl_list, delta=0.1)
        elif name == 'Ensemble_VR':
            return Ensemble_VR(X_tr, Y_tr, idxs_lb, net, handler, args, seeds=[1, 12, 123, 1234, 12345])
        elif name == 'WAAL':
            return WAAL
        elif name == 'NTKSampling':
            return NTKSampling(X_tr, Y_tr, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)
        elif name == 'NTK_Clsuter':
            return NTK_Clsuter(X_tr, Y_tr, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)
        elif name == 'EntropySamplingDouble':
            return EntropySamplingDouble(X_tr, Y_tr, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)
        elif name == 'KGradSampling':
            return KGradSampling(X_tr, Y_tr, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)
        elif name == 'GNormSampling':
            return GNormSampling(X_tr, Y_tr, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)
        elif name == 'HistoryEntropySampling':
            return HistoryEntropySampling(X_tr, Y_tr, idxs_lb, net, handler, args, X_val=X_val, Y_val=Y_val)
        else:
            raise NotImplementedError
    strategy = get_strategy(STRATEGY_NAME)

    # store accuracies 
    accuracy = []

    # print info
    print(args.data)
    print('SEED {}'.format(args.seed))
    print(type(strategy).__name__)

    # round 0 accuracy
    strategy.new_selected = np.where(idxs_lb != 0)[0]
    strategy.train()

    P = strategy.predict(X_te, Y_te)
    acc = np.zeros(args.num_round+1)
    acc[0] = 1.0 * (Y_te==P).sum().item() / len(Y_te)
    print('Round 0\ntesting accuracy {}'.format(acc[0]))
    class_entropy = []
    class_dist = []

    for rd in range(1, int(NUM_ROUND)+1):
        print('Round {}'.format(rd))

        # record time 
        start_time = time.time()

        strategy.estimate_fisher()
        strategy.estimate_z()
        # strategy.estimate_mas()

        q_idxs = strategy.query(args.num_query, pool_size=5000)
        # q_idxs, p_idxs = strategy.query(args.num_query, pool_size=10000)
        idxs_lb[q_idxs] = True
        
        # print_points(strategy, q_idxs)
        strategy.new_selected = q_idxs
        # strategy.new_selected_bottom = p_idxs
        
        cls_dist = strategy.Y[q_idxs].bincount().numpy() / len(q_idxs)
        class_entropy.append(np.sum(-cls_dist * np.log(cls_dist + 1e-5)))
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
        end_time = time.time()
        # formulate output 
        output['model'].append(STRATEGY_NAME)
        output['round'].append(rd)
        output['time'].append(end_time-start_time)
        output['accuracy'].append(acc[rd])
        output['size'].append(int(NUM_INIT_LB)+int(NUM_QUERY)*int(NUM_ROUND))
        output['dataset'].append(DATA_NAME)
        output['num_init_lb'].append(NUM_INIT_LB)
        output['num_query'].append(NUM_QUERY)
        output['num_round'].append(NUM_ROUND)
        output['strategy'].append(STRATEGY_NAME)
        if args.continual == True: output['train'].append('Continual')
        else: output['train'].append('Full')

    # print results
    print('SEED {}'.format(args.seed))
    print(type(strategy).__name__)
    print(','.join([str(a) for a in acc]))
    print('NUM Query: ', args.num_query)
    print('Class entropy: ', class_entropy)

# store output 
df = pd.DataFrame(data=output)
today = date.today().strftime('%m%d')
if args.continual == True: 
    path = f"{DATA_NAME}_{NUM_INIT_LB}_{NUM_QUERY}_{NUM_ROUND}_cont_{today}_lwf.csv"
else:
    path = f"{DATA_NAME}_{NUM_INIT_LB}_{NUM_QUERY}_{NUM_ROUND}_full_{today}.csv"
# df.to_csv('/u/yh9vhg/deep-active-learning/training_output/cifar.csv')
df.to_csv(f'/u/yh9vhg/CAL/training_output/{path}')
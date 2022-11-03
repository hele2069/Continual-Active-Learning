from collections import defaultdict
from backpack.extensions.backprop_extension import FAIL_ERROR
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import IPython
from copy import deepcopy
import torch.nn as nn
from backpack import backpack
from backpack.extensions import BatchGrad
from backpack import extend
import IPython
import matplotlib.pyplot as plt
from collections import Counter
from multiprocessing import reduction
from telnetlib import IP
from xmlrpc.client import boolean

class Strategy(nn.Module):
    def __init__(self, X, Y, idxs_lb, net, handler, args, seeds=None, X_val=None, Y_val=None):
        super().__init__()
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.net = net
        self.handler = handler
        self.args = args
        self.continual = args.continual # continual learning switch
        self.n_pool = len(Y)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.seeds = seeds

        self.X_val = X_val
        self.Y_val = Y_val
        self.pred_history = np.zeros((len(Y), 10))
        self.new_selected = None
        self.new_selected_bottom = None
        
        self.clf = extend(self.net().to(self.device))
        self.clf_init = deepcopy(self.clf)
        self.round = 0

        # EWC
        self.ewc_lambda = args.ewc_lambda
        self.gamma = 0.9
        self.online_ewc = args.online_ewc
        self.fisher_n = None
        self.emp_FI = True
        self.EWC_task_count = 0
        self.EWC_eps = 1e-20
        self.lambda_ = 0.
        
        self.ucb_lambda = 1e-4
        self.z = {}

        self.cls_mean = []
        self.uncertainity = []
        self.z_history = {}

    def query(self, n, pool_size):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb
        self.round += 1
        # if self.round > 5:
        #    self.ucb_lambda = 1e-5

    def _train(self, epoch, loader_tr, optimizer, loader_bottom=None):
        self.clf.train()
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out = self.clf(x)

            y_onehot = torch.FloatTensor(len(y), self.args.n_class).to(self.device)
            y_onehot.zero_()
            y_onehot.scatter_(1, y[:, None], 1)
            loss = F.cross_entropy(out, y)

            if self.round > 0 and self.continual:
                '''
                x = next(iter(loader_bottom))[0]
                x = x.to(self.device)
                y_prob = self.last_clf(x).softmax(1).detach()
                out = self.clf(x)
                out_logprob = torch.log_softmax(out, 1)
                loss = (1 - self.lambda_) * loss + self.lambda_ * -torch.mean(torch.sum(y_prob * out_logprob, 1))
                '''
                
                '''
                y_prob = self.last_clf(x).softmax(1).detach()
                lambda_ = self.lambda_
                y = lambda_ * y_prob + (1-lambda_) * y_onehot
                out_logprob = torch.log_softmax(out, 1)
                loss = -torch.mean(torch.sum(y * out_logprob, 1))
                '''
                ewc_loss = self.ewc_lambda * self.ewc_loss()    
                loss += ewc_loss
            loss.backward()
            optimizer.step()
    
    def _train_noisy(self, epoch, loader_tr, optimizer, ins_weight):
        self.clf.train()
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out = self.clf(x)
            # loss = F.cross_entropy(out, y)
            loss = F.cross_entropy(out, y, reduction='none')
            loss = torch.mean(loss * ins_weight[idxs])
            loss.backward()
            optimizer.step()

    def train(self, use_history=False):
        n_epoch = self.args.n_epoch
        if self.continual:
            idxs_train = self.new_selected 
        else:
            self.clf = deepcopy(self.clf_init)
            idxs_train = np.arange(self.n_pool)[self.idxs_lb]

        self.total_param = sum(p.numel() for p in self.clf.parameters() if p.requires_grad)
        self.best_clf = None
        self.val_acc = 0
        optimizer = optim.SGD(self.clf.parameters(), **self.args.optimizer_args)

        loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train], 
                            transform=self.args.transform, total_size=None),
                            shuffle=True, **self.args.loader_tr_args)

        loader_bottom = DataLoader(self.handler(self.X[self.new_selected_bottom], 
                            self.Y[self.new_selected_bottom], 
                            transform=self.args.transform, total_size=None),
                            shuffle=True, **self.args.loader_tr_args)
        for epoch in range(1, n_epoch + 1):
            if epoch > 70:
                optimizer = optim.SGD(self.clf.parameters(), lr=0.001, momentum=0.5)
            if not use_history:
                self._train(epoch, loader_tr, optimizer, loader_bottom=loader_bottom)
            else:
                print('history')
                self._train_noisy(epoch, loader_tr, optimizer, ins_weight)
            val_acc = self.validation()
            print('Epoch [%2d] valid acc: %.4f' % (epoch, val_acc))

        self.last_clf = deepcopy(self.clf)
    
    def train_ensemble(self):
        n_epoch = self.args.n_epoch
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train], transform=self.args.transform),
                            shuffle=True, **self.args.loader_tr_args)
        self.clfs = []
        for s in self.seeds:
            print('Model [%d] is training' % s)
            torch.manual_seed(s)
            torch.cuda.manual_seed(s)
            
            self.clf = self.net().to(self.device)
            optimizer = optim.SGD(self.clf.parameters(), **self.args.optimizer_args)
            for epoch in range(1, n_epoch+1):
                self._train(epoch, loader_tr, optimizer)
                self.validation()
            self.clfs.append(self.clf)

    def predict_ensemble(self, X, Y, mode='train'):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args.transform),
                            shuffle=False, **self.args.loader_te_args)
        P = torch.zeros((len(self.seeds), len(Y)), dtype=Y.dtype)
        prob = np.zeros((len(Y), 10))
        for i, clf in enumerate(self.clfs):
            clf.eval()
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out = clf(x)

                pred = out.max(1)[1]
                P[i][idxs] = pred.cpu()
        return P      

    def predict(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args.transform),
                            shuffle=False, **self.args.loader_te_args)
        self.best_clf.eval()
        P = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out = self.best_clf(x)

                pred = out.max(1)[1]
                P[idxs] = pred.cpu()
        return P
    
    def validation(self):
        loader_val = DataLoader(self.handler(self.X_val, self.Y_val, 
                                transform=self.args.transform), batch_size=1024, shuffle=False)
        self.clf.eval()
        correct = 0
        with torch.no_grad():
            for x, y, idxs in loader_val:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                pred = out.max(1)[1]
                correct += pred.eq(y).sum().item()
        val_acc = correct / len(self.Y_val)
        if val_acc >= self.val_acc:
            self.best_clf = deepcopy(self.clf)
            self.val_acc = val_acc
        return val_acc

    def predict_prob(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args.transform),
                            shuffle=False, **self.args.loader_te_args)

        self.clf.eval()
        probs = torch.zeros([len(Y), self.args.n_class])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs

    def predict_prob_dropout(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args.transform),
                            shuffle=False, **self.args.loader_te_args)

        self.clf.train()
        probs = torch.zeros([len(Y), self.args.n_class])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i+1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop 
        return probs

    def predict_prob_dropout_split(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args.transform),
                            shuffle=False, **self.args.loader_te_args)

        self.clf.train()
        probs = torch.zeros([n_drop, len(Y), len(np.unique(Y))])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i+1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.clf(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        
        return probs

    def get_embedding(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args.transform),
                            shuffle=False, **self.args.loader_te_args)

        self.clf.eval()
        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                e1 = self.clf.get_embedding(x)
                embedding[idxs] = e1.cpu()
        return embedding

    def params_mask(self, array, mask_rate):
        def _topk(array, k):
            v, i = torch.topk(array.flatten(), k)
            return np.array(np.unravel_index(i.numpy(), array.shape)).T

        num_masks = int(array.flatten().size()[0] * mask_rate)
        mask_idxes = _topk(-array, num_masks)
        array_mask = np.zeros_like(array, dtype=bool)

        if len(array.shape) == 2:
            array_mask[mask_idxes[:, 0], mask_idxes[:, 1]] = True
        else:
            array_mask[mask_idxes[:, 0]] = True
        return array_mask

    def estimate_fisher(self):
        self.clf.eval()
        est_fisher_info = {}
        for n, p in self.clf.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()

        if self.online_ewc:
            idxs_train = self.new_selected
        else:
            idxs_train = np.arange(self.n_pool)[self.idxs_lb]

        loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train], 
                                transform=self.args.transform, total_size=None),
                                shuffle=False, **self.args.loader_te_args)        

        for index, (x, y, idxs) in enumerate(loader_tr):
            if self.fisher_n is not None:
                if index >= self.fisher_n:
                    break
            x = x.to(self.device)
            output = self.clf(x)
            if self.emp_FI:
                label = y.to(self.device)
            else:
                label = output.max(1)[1]

            outsoft = output.softmax(1)
            sum_out = torch.sum(outsoft)
            nlloss = F.nll_loss(F.log_softmax(output, dim=1), label, reduction='sum')
            with backpack(BatchGrad()):
                nlloss.backward()
                # sum_out.backward()
            for n, p in self.clf.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if hasattr(p, 'grad_batch') and p.grad_batch is not None:
                        est_fisher_info[n] += torch.sum(p.grad_batch.detach() ** 2, 0)

        est_fisher_info = {n: p / len(idxs_train) for n, p in est_fisher_info.items()}
        for n, p in self.clf.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.register_buffer('{}_EWC_prev_task'.format(n, ''), p.detach().clone())
                if self.online_ewc and self.EWC_task_count > 0:
                    existing_values = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                    est_fisher_info[n] += self.gamma * existing_values

                fisher_param_mask = self.params_mask(est_fisher_info[n].cpu(), 0.2)

                self.register_buffer('{}_NTK_fisher_mask'.format(n), 
                                    torch.from_numpy(fisher_param_mask))

                self.register_buffer('{}_EWC_estimated_fisher'.format(n), 
                                    est_fisher_info[n])
        self.EWC_task_count = 1 if self.online_ewc else self.EWC_task_count + 1

    def ewc_loss(self):
        if self.EWC_task_count > 0:
            losses = []
            fisher_max = 0
            fisher_min = 1000
            for n, p in self.clf.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    mean = getattr(self, '{}_EWC_prev_task{}'.format(n, ""))
                    fisher = getattr(self, '{}_EWC_estimated_fisher{}'.format(n, ""))
                    if fisher.max() > fisher_max:
                        fisher_max = fisher.max()
                    if fisher.min() < fisher_min:
                        fisher_min = fisher.min()

            for n, p in self.clf.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    mean = getattr(self, '{}_EWC_prev_task{}'.format(n, ""))
                    fisher = getattr(self, '{}_EWC_estimated_fisher{}'.format(n, ""))

                    fisher = (fisher - fisher.min()) / (fisher.max() - fisher.min())
                    losses.append((fisher * (p-mean)**2).sum())

            return (1./2) * sum(losses)
        else:
            return torch.tensor(0, device=self.device)
    
    def estimate_ntk(self, X, Y, embed=False):
        self.clf.eval()
        loader_te = DataLoader(self.handler(X, Y, transform=self.args.transform),
                                shuffle=False, **self.args.loader_te_args)
        
        cb = np.zeros(len(Y))
        ent = np.zeros(len(Y))
        expectation = False
        cls_prob = np.zeros(self.args.n_class)
        
        cb_history = defaultdict(list)
        deno_history = defaultdict(list)

        for index, (x, y, idxs) in enumerate(loader_te):
            x = x.to(self.device)
            output = self.clf(x)
            # output = output.softmax(1)
            sum_out = torch.sum(output)
            if not expectation:
                label = output.max(1)[1]
                # sum_out = F.nll_loss(F.log_softmax(output, dim=1), label)
                sum_out = F.cross_entropy(output, label, reduction='sum')
                
                with backpack(BatchGrad()):
                    sum_out.backward()
                
                for n, p in list(self.clf.named_parameters()):
                    n = n.replace('.', '__')
                    if p.grad_batch is not None:
                        p_grad = p.grad_batch.detach()
                        p_grad = p_grad.flatten(start_dim=1) ** 2
                        # p_grad = p_grad / (p_grad.max(1)[0] 
                        #                              - p_grad.min(1)[0] + self.EWC_eps)[:, None]
                        deno = self.z[n].flatten(start_dim=0)
                        deno_cls = self.z_cls[n].flatten(start_dim=1)
                        
                        # cb_raw = p_grad.flatten(start_dim=1)[:, deno == 0]
                        cb_raw = p_grad.flatten(start_dim=1) / deno
                        cb_raw_cls = p_grad.flatten(start_dim=1)[:, None] / deno_cls

                        # cb_value, cb_index = cb_raw.sort()[0][:, -100:].detach().cpu().numpy(), cb_raw.sort()[1][:, -100:].detach().cpu().numpy()
                        cb_history[n].extend(cb_raw.detach().cpu().numpy())
                        # cb_history[n+'_index.extend(cb_index)

                        cb_mask = getattr(self, '{}_NTK_fisher_mask'.format(n)).flatten()

                        use_mask = True
                        if use_mask:
                            cb_raw = cb_raw[:, cb_mask]

                        cb_raw = torch.sum(cb_raw, 1).detach().cpu().numpy()
                        cb[idxs] += cb_raw
                
                # prob = output.softmax(1).max(1)[0]
                # cb[idxs] = (1 - prob).detach().cpu().numpy()
                output = output.softmax(1)
                ent[idxs] = -torch.sum(output * torch.log(output), 1).detach().cpu().numpy()
            else:
                prob = output.softmax(1)
                prob[prob < 1e-3] = 0
                for label in range(self.args.n_class):
                    output = self.clf(x)
                    label_vec = torch.ones(len(y)).to(self.device) * label
                    sum_out = F.cross_entropy(output, label_vec.long(), reduction='sum')

                    self.clf.zero_grad()
                    with backpack(BatchGrad()):
                        sum_out.backward()

                    for n, p in self.clf.named_parameters():
                        n = n.replace('.', '__')
                        if p.grad_batch is not None:
                            p_grad = p.grad_batch.detach()
                            p_grad = p_grad.flatten(start_dim=1) ** 2
                            # deno = self.z[n].flatten(start_dim=0)

                            deno = self.z_cls[n].flatten(start_dim=1)[label]
                            cb[idxs] += prob[:, label].detach().cpu().numpy() * torch.sum(p_grad.flatten(start_dim=1) / deno , 1).detach().cpu().numpy()

                            # cb[idxs] += prob[:, label].detach().cpu().numpy() * torch.sum(p_grad.flatten(start_dim=1) / (self.z[n].flatten(start_dim=0)
                            #                 + 1), 1).detach().cpu().numpy()
        
       
        cb_vec = []
        deno_vec = []
        for k in cb_history:
            deno = self.z[k].flatten(start_dim=0)
            cb_history[k] = np.stack(cb_history[k], 0)
            deno_vec.append((1 / deno).detach().cpu().numpy())
            cb_vec.append(cb_history[k])
        self.cb_vec = np.concatenate(cb_vec, 1)
        deno_vec = np.concatenate(deno_vec, 0)
        
        # cb_idx = cb_vec.argsort()
        # cb_idx = cb_idx[:, -100:]
        
        '''
        cb_dist = np.bincount(cb_idx.flatten())
        # a = [str(i) for i in cb_dist.argsort()]
        plt.bar(np.arange(len(cb_dist)), cb_dist)
        # plt.plot(cb_dist)
        plt.savefig('./results/{}/cb_dist_{}.png'.format(self.data, self.round))
        plt.clf()
        '''

        self.cls_mean.append(cls_prob / len(Y))
        score = np.sqrt(cb)
        #cb = np.sqrt(cb) / np.sum(np.sqrt(cb))
        # ent = ent / np.sum(ent)
        # score = cb * ent
        return score

    def estimate_z(self):
        self.clf.eval()       
        # self.clf_init.eval()

        # idxs_train = self.new_selected
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train], transform=self.args.transform, total_size=None),
                            shuffle=False, **self.args.loader_te_args)
        tmp_z = {}
        tmp_z_cls = {}
        for index, (x, y, idxs) in enumerate(loader_tr):
            x = x.to(self.device)
            output = self.clf(x)
            # output = self.clf_init(x)
            # output = output.softmax(1)
            sum_out = torch.sum(output)

            label = y.to(self.device)
            label_prob = torch.gather(output, index=label[:, None], dim=1).squeeze()
            sum_out = F.cross_entropy(output, label, reduction='sum')
            # sum_out = F.nll_loss(F.log_softmax(output, dim=1), label, reduction='sum')
            
            with backpack(BatchGrad()):
                sum_out.backward()

            for n, p in list(self.clf.named_parameters()):
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if hasattr(p, 'grad_batch') and p.grad_batch is not None:
                        p_grad = p.grad_batch.detach()
                        if n not in tmp_z:
                            tmp_z[n] = torch.sum((p_grad ** 2).flatten(start_dim=1), 0)
                            tmp_z_cls[n] = torch.stack([torch.sum((p_grad ** 2).flatten(start_dim=1)[y == i], 0) for i in range(10)], 0)

                        else:
                            tmp_z[n] += torch.sum((p_grad ** 2).flatten(start_dim=1), 0)
                            tmp_z_cls[n] += torch.stack([torch.sum((p_grad ** 2).flatten(start_dim=1)[y == i], 0) for i in range(10)], 0)

        self.z = {}
        self.z_cls = {}
        for n, p in list(self.clf.named_parameters()):
            if p.requires_grad:
                n = n.replace('.', '__')
                
                if n not in tmp_z:
                    continue

                if n not in self.z:
                    self.z[n] = tmp_z[n] + self.ucb_lambda
                    self.z_cls[n] = tmp_z_cls[n] + self.ucb_lambda
                else:
                    self.z[n] = self.z[n] + tmp_z[n]
                    self.z_cls[n] = self.z_cls[n] + tmp_z_cls[n]
                # self.z[n] = (tmp_z[n] - tmp_z[n].min()) / ((tmp_z[n].max() - tmp_z[n].min()) + self.EWC_eps)
                # self.z[n] = (self.z[n] - self.z[n].min()) / ((self.z[n].max() - self.z[n].min()) + self.EWC_eps)
                if n +'_mean' not in self.z_history:
                    self.z_history[n+'_mean'] = [np.mean((1 / self.z[n]).cpu().numpy())]
                    self.z_history[n+'_std'] = [np.std((1 / self.z[n]).cpu().numpy())]
                elif n +'_mean' in self.z_history:
                    self.z_history[n+'_mean'].append(np.mean((1 / self.z[n]).cpu().numpy()))
                    self.z_history[n+'_std'].append(np.std((1 / self.z[n]).cpu().numpy()))

        # self.z_history.append(self.z)
    
    '''
    def get_grad_embedding(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args.transform']),
                            shuffle=False, **self.args.loader_te_args'])

        self.clf.eval()
        embedding = []
        for x, y, idxs in loader_te:
            x, y = x.to(self.device), y.to(self.device)
                
            self.clf.zero_grad()
            output = self.clf(x)
            # output = output.softmax(1)
            # sum_out = torch.sum(output)

            # label = output.max(1)[1]
            # sum_out = F.nll_loss(F.log_softmax(output, dim=1), label)
            with backpack(BatchGrad()):
                sum_out.backward()
            g_list = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.clf.parameters() if p.requires_grad], dim=1).cpu().numpy()
            embedding.extend(g_list)
        return np.array(embedding)
    '''

    def get_grad_embedding(self, X, Y):
        model = self.clf
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = len(np.unique(Y))
        embedding = np.zeros([len(Y), embDim])
        loader_te = DataLoader(self.handler(X, Y, transform=self.args.transform),
                            shuffle=False, **self.args.loader_te_args)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x = x.to(self.device)
                cout= self.clf(x)
                out = self.clf.get_embedding(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1)
                maxInds = torch.argmax(batchProbs,1)
                score = torch.gather(batchProbs, index=maxInds[:, None], dim=1).squeeze()
                score = score.cpu().numpy()
                embedding[idxs] = out * (1 - score)[:, None]
            return torch.Tensor(embedding)
    
    


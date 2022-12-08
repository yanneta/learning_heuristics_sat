import argparse
import logging
import os
import pdb
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from cnf import CNF
from local_search import WalkSATLN
from warm_up import WarmUP

class Net(nn.Module):
    def __init__(self, input_features=4, hidden=10):
        super(Net, self).__init__()
        self.lin = nn.Linear(input_features, hidden)
        self.dropout = nn.Dropout(0.3)
        self.lin2 = nn.Linear(hidden, 1)
    
    def forward(self, x):
        x = self.lin(x)
        x = F.relu(self.dropout(x))
        x = self.lin2(x)
        return x

class NoiseNet(nn.Module):
    def __init__(self, input_features=1):
        super(NoiseNet, self).__init__()
        self.lin = nn.Linear(input_features, 1)
        self.lin.weight.data.uniform_(1, 1.5)
        self.lin.bias.data.fill_(0.5)
    def forward(self, x):
        x = self.lin(x)
        return 0.5*torch.sigmoid(x)

class Net2(nn.Module):
    def __init__(self, input_features=4):
        super(Net2, self).__init__()
        self.lin = nn.Linear(input_features, 1)
    def forward(self, x):
        x = self.lin(x)
        return x

def load_dir(path):
    data = []
    for filename in os.listdir(path):
        name, ext = os.path.splitext(filename)
        if ext != '.cnf':
            continue
        f = CNF.from_file(os.path.join(path, filename))
        data.append(f)
    return data

def split_data(data, num_vals=100):
    logging.info("length of data is " + str(len(data)))
    N = len(data) - num_vals
    train_ds = data[:N]
    val_ds = data[N:]
    return train_ds, val_ds

def change_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr

def compute_mean_median_CI(values):
    N = len(values)
    medians = [np.median(np.random.choice(values, N)) for i in range(1000)]
    means = [np.mean(np.random.choice(values, N)) for i in range(1000)]
    return np.quantile(means, q=[.25, .975]), np.quantile(medians, q=[.25, .975])

def compute_median_per_obs(flips, max_tries):
    return [np.median(flips[i:i+max_tries]) for i in range(len(flips)//max_tries)]

def to_log(flips, backflips,  loss, accuracy, comment, CI=False, max_tries=None):
    """when max_tries is not None we compute median flips per observation"""
    if loss is None:
        loss = -1
    if max_tries:
        med_flips = compute_median_per_obs(flips, max_tries)
        med_backflips = compute_median_per_obs(backflips, max_tries)
    formatting = '{} Flips Med: {:.2f}, Mean: {:.2f} Backflips Med: {:.2f} Mean: {:.2f} Acc: {:.2f} Loss: {:.2f}'
    text = formatting.format(comment, np.median(flips), np.mean(flips), np.median(backflips), \
        np.mean(backflips), 100 * accuracy, loss)
    logging.info(text)
    if CI:
        ci_means, ci_median = compute_mean_median_CI(flips)
        formatting = 'CI means FLIPS ({:.2f}, {:.2f}), CI median ({:.2f}, {:.2f})'
        text = formatting.format(ci_means[0], ci_means[1], ci_median[0], ci_median[0])
        logging.info(text)
        ci_means, ci_median = compute_mean_median_CI(backflips)
        formatting = 'CI means BACKFLIPS ({:.2f}, {:.2f}), CI median ({:.2f}, {:.2f})'
        text = formatting.format(ci_means[0], ci_means[1], ci_median[0], ci_median[0])
        logging.info(text)


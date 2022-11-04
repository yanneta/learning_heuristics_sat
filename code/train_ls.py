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
    def __init__(self, input_features=4, hidden=5):
        super(Net, self).__init__()
        self.lin = nn.Linear(input_features, hidden)
        self.dropout = nn.Dropout(0.5)
        self.lin2 = nn.Linear(hidden, 1)
    def forward(self, x):
        x = self.lin(x)
        x = F.relu(self.dropout(x))
        x = self.lin2(x)
        return x

def init_net(model):
    with torch.no_grad():
        model.lin.weight[0, 0] = 10
        model.lin2.weight[0, 0] = -1

class Net2(nn.Module):
    def __init__(self, input_features=4):
        super(Net2, self).__init__()
        self.lin = nn.Linear(input_features, 1)
    def forward(self, x):
        x = self.lin(x)
        return x

def init_net2(policy, input_features=4):
    with torch.no_grad():
        policy.lin.weight[0, 0] = -1
        for i in range(1, input_features):
            policy.lin.weight[0, i] = 0
        policy.lin.bias[0] = 0


def load_dir(path):
    data = []
    for filename in os.listdir(path):
        name, ext = os.path.splitext(filename)
        if ext != '.cnf':
            continue
        f = CNF.from_file(os.path.join(path, filename))
        data.append(f)
    return data

def split_data(data):
    logging.info("length of data is " + str(len(data)))
    train_ds = data[:1500]
    val_ds = data[1500:1700]
    test_ds = data[1700:]
    return train_ds, val_ds, test_ds

def compute_mean_median_CI(values):
    N = len(values)
    medians = [np.median(np.random.choice(values, N)) for i in range(1000)]
    means = [np.mean(np.random.choice(values, N)) for i in range(1000)]
    return np.quantile(means, q=[.25, .975]), np.quantile(medians, q=[.25, .975])

def to_log(flips, backflips,  loss, accuracy, comment, CI=False):
    if loss is None:
        loss = -1
    text = '{} Flips Med: {:.2f}, Mean: {:.2f} Backflips Med: {:.2f} Mean: {:.2f} Acc: {:.2f} Loss: {:.2f}'.format(
            comment, np.median(flips), np.mean(flips), np.median(backflips), np.mean(backflips), 100 * accuracy, loss)
    logging.info(text)
    if CI:
        ci_means, ci_median = compute_mean_median_CI(flips)
        text = 'CI means FLIPS ({:.2f}, {:.2f}), CI median ({:.2f}, {:.2f})'.format(ci_means[0], ci_means[1], ci_median[0], ci_median[0])
        logging.info(text)
        ci_means, ci_median = compute_mean_median_CI(backflips)
        text = 'CI means BACKFLIPS ({:.2f}, {:.2f}), CI median ({:.2f}, {:.2f})'.format(ci_means[0], ci_means[1], ci_median[0], ci_median[0])
        logging.info(text)


def main(args):
    if args.seed > -1:
        random.seed(args.seed)

    basename = args.dir_path.replace("../", "").replace("/","_") + "_d_" +  str(args.discount) 
    basename += "_e" + str(args.epochs)
    if args.warm_up:
         basename += "_wup"
    log_file = "logs/" + basename +  ".log"
    model_file = "models/" + basename + ".pt" 
    print(log_file)

    logging.basicConfig(filename=log_file, level=logging.INFO)

    data = load_dir(args.dir_path)
    train_ds, val_ds, test_ds = split_data(data)

    policy = Net()
    optimizer = optim.RMSprop(policy.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.warm_up:
        wup = WarmUP(policy)
        for i in range(5):
            wup.train_epoch(optimizer, train_ds)


    ls = WalkSATLN(policy, args.max_tries, args.max_flips, discount=args.discount)
    flips, backflips,  loss, accuracy = ls.evaluate(val_ds, walksat=True)
    to_log(flips, backflips,  loss, accuracy, comment="EVAL Walksat")
    flips, backflips,  loss, accuracy = ls.evaluate(val_ds)
    to_log(flips, backflips,  loss, accuracy, comment="EVAL No Train")
    best_median_flips = np.median(flips)
    best_epoch = 0
    torch.save(policy.state_dict(), model_file)
    for i in range(1, args.epochs + 1):
        print("epoch ", i)
        flips, backflips, loss, accuracy = ls.train_epoch(optimizer, train_ds)
        to_log(flips, backflips,  loss, accuracy, comment="Train Ep " + str(i))
        flips, backflips, loss, accuracy = ls.evaluate(val_ds)
        if i%5 == 0 and i > 0:
            to_log(flips, backflips,  loss, accuracy, comment="EVAL  Ep " + str(i))
            if best_median_flips > np.median(flips):
                torch.save(policy.state_dict(), model_file)
                best_median_flips = np.median(flips)
                best_epoch = i
    # Test
    ls.policy.load_state_dict(torch.load(model_file))
    flips, backflips,  loss, accuracy = ls.evaluate(test_ds)
    to_log(flips, backflips,  loss, accuracy, comment="TEST", CI=True)
    logging.info("Best epoch " + str(best_epoch))
    print("Best epoch", best_epoch)
   
    flips, backflips,  loss, accuracy = ls.evaluate(test_ds, walksat=True)
    to_log(flips, backflips,  loss, accuracy, comment="TEST Walksat", CI=True)
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir_path', type=str)
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_tries', type=int, default=10)
    parser.add_argument('--max_flips', type=int, default=10000)
    parser.add_argument('--p', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--discount', type=float, default=0.5)
    parser.add_argument('--warm_up', type=bool, default=True)
    args = parser.parse_args()
    main(args)

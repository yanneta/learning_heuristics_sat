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
from utils import *


def train_policy(ls, optimizer, train_ds, val_ds, args, best_median_flips, model_file):
    best_epoch = 0
    torch.save(ls.policy.state_dict(), model_file)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=1, epochs=args.epochs,
        div_factor=5, final_div_factor=10)
    for i in range(1, args.epochs + 1):
        print("epoch ", i)
        flips, backflips, loss, accuracy = ls.train_epoch(optimizer, train_ds)
        to_log(flips, backflips,  loss, accuracy, comment="Train Ep " + str(i))
        flips, backflips, loss, accuracy = ls.evaluate(val_ds)
        scheduler.step()
        if i%5 == 0 and i > 0:
            to_log(flips, backflips,  loss, accuracy, "EVAL  Ep " + str(i), args.max_tries)
            med_flips = compute_median_per_obs(flips, args.max_tries)
            if best_median_flips > np.median(med_flips):
                torch.save(ls.policy.state_dict(), model_file)
                best_median_flips = np.median(med_flips)
                best_epoch = i
    formatting = 'Best Flips Med: {:.2f}, Best epoch: {}'
    text = formatting.format(best_median_flips,  best_epoch)
    logging.info(text)

def train_warm_up(policy, optimizer, train_ds, max_flips=5000):
    wup = WarmUP(policy, max_flips=max_flips)
    for i in range(args.warm_up):
        loss = wup.train_epoch(optimizer, train_ds)
        logging.info('Warm_up train loss {:.2f}'.format(loss))

def create_filenames(args):
    basename = args.dir_path.replace("../", "").replace("/", "_") + "_d_" +  str(args.discount)
    basename += "_e" + str(args.epochs) + "_cos_lr" + "p_" +  str(args.p) + "_normv" + "_dos"
    if args.warm_up > 0:
         basename += "_wup"
    log_file = "logs/" + basename +  ".log"
    model_file = "models/" + basename +  "v3.pt"
    return log_file, model_file

def main(args):
    if args.seed > -1:
        random.seed(args.seed)

    log_file, model_file = create_filenames(args)
    print(log_file)

    logging.basicConfig(filename=log_file, level=logging.INFO)

    data = load_dir(args.dir_path)
    train_ds, val_ds = split_data(data)

    policy = Net2(input_features=5)
    optimizer = optim.AdamW(policy.parameters(), lr=args.lr/3, weight_decay=1e-5)

    if args.warm_up > 0:
        train_warm_up(policy, optimizer, train_ds)

    print("Eval WalkSAT")
    ls = WalkSATLN(policy, args.max_tries, args.max_flips, discount=args.discount)
    flips, backflips,  loss, accuracy = ls.evaluate(val_ds, walksat=True)
    to_log(flips, backflips,  loss, accuracy, "EVAL Walksat", args.max_tries)
    flips, backflips,  loss, accuracy = ls.evaluate(val_ds)
    to_log(flips, backflips,  loss, accuracy, "EVAL No Train/ WarmUP", args.max_tries)
    best_median_flips = np.median(flips)

    train_policy(ls, optimizer, train_ds, val_ds, args, best_median_flips, model_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir_path', type=str)
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_tries', type=int, default=10)
    parser.add_argument('--max_flips', type=int, default=10000)
    parser.add_argument('--p', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--discount', type=float, default=0.5)
    parser.add_argument('--warm_up', type=int, default=10)
    args = parser.parse_args()
    main(args)

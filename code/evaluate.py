import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#from cnf import CNF
from local_search import WalkSATLN
from utils import *

def main(args):
    if args.seed > -1:
        random.seed(args.seed)

    logging.basicConfig(filename=log_file, level=logging.INFO)

    data = load_dir(args.dir_path)
    policy = Net2(input_features=5)
    ls = WalkSATLN(policy, args.max_tries, args.max_flips)
    ls.policy.load_state_dict(torch.load(model_file))
    flips, backflips,  loss, accuracy = ls.evaluate(data)
    to_log(flips, backflips,  loss, accuracy, "TEST", True, args.max_tries)
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir_path', type=str)
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_tries', type=int, default=10)
    parser.add_argument('--max_flips', type=int, default=10000)
    parser.add_argument('--p', type=float, default=0.5)
    args = parser.parse_args()
    main(args)

import argparse
import logging
import os
import random
from pathlib import Path, PurePath

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from local_search import WalkSATLN
from utils import *

def main(args):
    if args.seed > -1:
        random.seed(args.seed)
    basename = "test_" + PurePath(args.dir_path).name 
    basename += Path(args.model_path).stem 
    log_file = "logs/" + basename +  ".log"
    print(log_file)    
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info('Model path {}'.format(args.model_path))
    logging.info('Test data path {}'.format(args.dir_path))
    data = load_dir(args.dir_path)[:2]
    policy = Net2(input_features=5)
    noise_policy = NoiseNet()

    ls = WalkSATLN(policy, noise_policy, args.max_tries, args.max_flips)
    ls.policy.load_state_dict(torch.load(args.model_path))
    ls.noise_policy.load_state_dict(torch.load(args.noise_model_path))
    flips, backflips,  loss, accuracy = ls.evaluate(data)
    to_log(flips, backflips,  loss, accuracy, "TEST", True, args.max_tries)
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir_path', type=str)
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('--noise_model_path', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_tries', type=int, default=10)
    parser.add_argument('--max_flips', type=int, default=10000)
    args = parser.parse_args()
    main(args)

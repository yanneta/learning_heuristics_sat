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
    p = get_p(args.dir_path)
    print(log_file)    
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info('Model path {}'.format(args.model_path))
    logging.info('Test data path {}'.format(args.dir_path))
    data = load_dir(args.dir_path)[:args.samples]
    policy = Net2(input_features=5)
    noise_policy = NoiseNet()
    policy.load_state_dict(torch.load(args.model_path))
    train_noise = False
    if args.noise_model_path is not None:
        noise_policy.load_state_dict(torch.load(args.noise_model_path))
        train_noise = True
    ls = WalkSATLN(policy, noise_policy, train_noise, args.max_tries, args.max_flips, p=p)
    med_flips, mean_flips,  accuracy = ls.evaluate(data)
    print("p is ", ls.p)
    to_log_eval(med_flips, mean_flips, accuracy, "TEST", True)
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir_path', type=str)
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('--noise_model_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_tries', type=int, default=10)
    parser.add_argument('--max_flips', type=int, default=10000)
    parser.add_argument('--samples', type=int, default=500)
    args = parser.parse_args()
    main(args)

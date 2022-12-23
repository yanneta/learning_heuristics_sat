#!/usr/bin/env python
# coding: utf-8

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from local_search import SATLearner


class WarmUP(SATLearner):
    def __init__(self, policy, noise_policy, max_flips=10000, p=0.5):
        super().__init__(policy, noise_policy, max_flips, p)
        self.break_histo = np.zeros(1000)
        
    def select_variable_reinforce(self, x, f, unsat_clause):
        index, lit = self.walksat_step(f, unsat_clause)
        logit = self.policy(x)
        log_prob = F.log_softmax(logit, dim=0)
        return index, log_prob[index]
    
    def reinforce_step(self, f, unsat_clause):
        x = self.stats_per_clause(f, unsat_clause)
        x = torch.from_numpy(x).float()
        index, log_prob = self.select_variable_reinforce(x, f, unsat_clause)
        literal = unsat_clause[index]
        return literal, log_prob
    
    def generate_episode_reinforce(self, f):
        self.sol = [x if random.random() < 0.5 else -x for x in range(f.n_variables + 1)]
        self.true_lit_count = self.compute_true_lit_count(f.clauses)
        self.age = np.zeros(f.n_variables + 1)
        self.age2 = np.zeros(f.n_variables + 1)
        log_probs = []
        self.flips = 0
        self.flipped = set()
        self.backflipped = 0
        while self.flips < self.max_flips:
            unsat_clause_indices = [k for k in range(len(f.clauses)) if self.true_lit_count[k] == 0]
            sat = not unsat_clause_indices
            if sat:
                break
            unsat_clause = f.clauses[random.choice(unsat_clause_indices)]
            self.flips +=1
            literal, log_prob, log_prob_p = self.select_literal(f, unsat_clause)           
            if log_prob:
                log_probs.append(-log_prob)
            self.update_stats(f, literal)
        loss = 0
        if len(log_probs) > 0:
            loss = torch.mean(torch.stack(log_probs))
        return sat, self.flips, self.backflipped, loss
    
    def evaluate(self, data):
        all_flips = []
        self.policy.eval()
        for f in data:
            sat, flips, backflipped, loss = self.generate_episode_reinforce(f)
            all_flips.append(flips)
        all_flips = np.array(all_flips).reshape(-1)
        return np.median(all_flips), np.mean(all_flips)
        
    def train_epoch(self, optimizer, data):
        losses = []
        flip_list = []
        for f in data:
            self.policy.train()
            sat, flips, backflipped, loss = self.generate_episode_reinforce(f)
            flip_list.append(flips) 
            if loss > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        return np.mean(losses), np.median(flip_list)



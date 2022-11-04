#!/usr/bin/env python
# coding: utf-8

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class WarmUP:
    def __init__(self, policy, max_flips=10000, p=0.5):
        self.policy = policy
        self.max_flips = max_flips
        self.p = p
        self.sol = []
        self.age = []
        self.last_10 = []
        self.break_histo = np.zeros(1000)
        
    def compute_true_lit_count(self, clauses):
        n_clauses = len(clauses)
        true_lit_count = [0] * n_clauses
        for index in range(n_clauses):
            for literal in clauses[index]:
                if self.sol[abs(literal)] == literal:
                    true_lit_count[index] += 1
        return true_lit_count

    def normalize_breaks(self, x):
        return np.minimum(x, 5)/5

    def do_flip(self, literal, occur_list):
        for i in occur_list[literal]:
            self.true_lit_count[i] += 1
        for i in occur_list[-literal]:
            self.true_lit_count[i] -= 1
        self.sol[abs(literal)] *= -1
        
    def stats_per_clause(self, f, unsat_clause):
        """ computes the featutes needed for the model
        """ 
        variables = [abs(v) for v in unsat_clause]
        breaks = np.zeros(len(variables))
        last_5 = self.last_10[:5]
        for i, literal in enumerate(unsat_clause):
            broken_count = 0
            for index in f.occur_list[-literal]:
                if self.true_lit_count[index] == 1:
                    broken_count += 1
            breaks[i] = broken_count
            self.break_histo[broken_count] +=1
        breaks = self.normalize_breaks(breaks)
        in_last_10 = np.array([int(v in self.last_10) for v in variables]) 
        age = np.array([self.age[v] for v in variables])/(self.age[0] + 1)
        in_last_5 = np.array([int(v in last_5) for v in variables]) 
        return np.stack([breaks, in_last_10, in_last_5, age], axis=1)
    
    def walksat_step(self, f, unsat_clause):
        """Returns chosen literal"""
        broken_min = float('inf')
        min_breaking_lits = []
        for i, literal in enumerate(unsat_clause):
            broken_count = 0
            for index in f.occur_list[-literal]:
                if self.true_lit_count[index] == 1:
                    broken_count += 1
                if broken_count > broken_min:
                    break
            if broken_count < broken_min:
                broken_min = broken_count
                min_breaking_lits = [i]
            elif broken_count == broken_min:
                min_breaking_lits.append(i)
        index = random.choice(min_breaking_lits)
        return index, unsat_clause[index]
    
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
        log_probs = []
        flips = 0
        flipped = set()
        backflipped = 0
        while flips < self.max_flips:
            unsat_clause_indices = [k for k in range(len(f.clauses)) if self.true_lit_count[k] == 0]
            sat = not unsat_clause_indices
            if sat:
                break
            unsat_clause = f.clauses[random.choice(unsat_clause_indices)]
            if random.random() < self.p:
                literal = random.choice(unsat_clause)
            else:
                literal, log_prob = self.reinforce_step(f, unsat_clause)
                log_probs.append(-log_prob)
            v = abs(literal)
            
            if v not in flipped:
                flipped.add(v)
            else:
                backflipped += 1
            self.last_10.insert(0, v)
            self.last_10 = self.last_10[:10]
            self.do_flip(literal, f.occur_list)
            flips += 1
            self.age[0] = flips
            self.age[v] = flips
        loss = 0
        if len(log_probs) > 0:
            loss = torch.mean(torch.stack(log_probs))
        return sat, flips, backflipped, loss
    
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
        for f in data:
            self.policy.train()
            sat, flips, backflipped, loss = self.generate_episode_reinforce(f)
            if loss > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        print(np.mean(losses))


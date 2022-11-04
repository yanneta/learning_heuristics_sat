#!/usr/bin/env python
# coding: utf-8

import os
from cnf import CNF
import numpy as np

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class WalkSATLN:
    def __init__(self, policy, max_tries=10, max_flips=1000, p=0.5, discount=0.5):
        self.policy = policy
        self.max_tries = max_tries
        self.max_flips = max_flips
        self.p = p
        self.discount = discount
        self.unsat_clauses = []
        self.age = []
        self.last_10 = []
        self.sol = []
        
    def compute_true_lit_count(self, clauses):
        n_clauses = len(clauses)
        true_lit_count = [0] * n_clauses
        for index in range(n_clauses):
            for literal in clauses[index]:
                if self.sol[abs(literal)] == literal:
                    true_lit_count[index] += 1
        return true_lit_count
    
    def select_variable_reinforce(self, x):
        logit = self.policy(x)
        prob = F.softmax(logit, dim=0)
        dist = Categorical(prob.view(-1))
        v = dist.sample()
        return v, dist.log_prob(v)
    
    def do_flip(self, literal, occur_list):
        for i in occur_list[literal]:
            self.true_lit_count[i] += 1
        for i in occur_list[-literal]:
            self.true_lit_count[i] -= 1
        self.sol[abs(literal)] *= -1

    def normalize_breaks(self, x):
        return np.minimum(x, 5)/5

    def stats_per_clause(self, f, unsat_clause):
        """ computes the featutes needed for the model
        """ 
        r = f.n_variables/ len(f.clauses)
        variables = [abs(v) for v in unsat_clause]
        breaks = np.zeros(len(variables))
        last_5 = self.last_10[:5]
        for i, literal in enumerate(unsat_clause):
            broken_count = 0
            for index in f.occur_list[-literal]:
                if self.true_lit_count[index] == 1:
                    broken_count += 1
            breaks[i] = broken_count
        breaks = self.normalize_breaks(breaks)
        in_last_10 = np.array([int(v in self.last_10) for v in variables]) 
        age = np.array([self.age[v] for v in variables])/(self.age[0] + 1)
        in_last_5 = np.array([int(v in last_5) for v in variables]) 
        return np.stack([breaks, in_last_10, in_last_5, age], axis=1)
    
    def walksat_step(self, f, unsat_clause):
        """Returns chosen literal"""
        broken_min = float('inf')
        min_breaking_lits = []
        for literal in unsat_clause:
            broken_count = 0
            for index in f.occur_list[-literal]:
                if self.true_lit_count[index] == 1:
                    broken_count += 1
                if broken_count > broken_min:
                    break
            if broken_count < broken_min:
                broken_min = broken_count
                min_breaking_lits = [literal]
            elif broken_count == broken_min:
                min_breaking_lits.append(literal)
        return random.choice(min_breaking_lits)
    
    def reinforce_step(self, f, unsat_clause):
        x = self.stats_per_clause(f, unsat_clause)
        x = torch.from_numpy(x).float()
        index, log_prob = self.select_variable_reinforce(x)
        literal = unsat_clause[index]
        return literal, log_prob
    
    def generate_episode_reinforce(self, f, walksat):
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
            log_prob = None
            if random.random() < self.p:
                literal = random.choice(unsat_clause)
            else:
                if walksat:
                    literal = self.walksat_step(f, unsat_clause)   
                else:
                    literal, log_prob = self.reinforce_step(f, unsat_clause)
                v = abs(literal)
                if v not in flipped:
                    flipped.add(v)
                else:
                    backflipped += 1
                self.last_10.insert(0, v)
                self.last_10 = self.last_10[:10]
                self.age[v] = flips
            self.do_flip(literal, f.occur_list)
            flips += 1
            self.age[0] = flips
            log_probs.append(log_prob)
        return sat, flips, backflipped, log_probs

    def reinforce_loss(self, log_probs_list):
        T = len(log_probs_list)
        log_probs_filtered = []
        mask = np.zeros(T, dtype=bool)
        for i, x in enumerate(log_probs_list):
            if x is not None:
                log_probs_filtered.append(x)
                mask[i] = 1

        log_probs = torch.stack(log_probs_filtered)
        p_rewards = self.discount ** torch.arange(T - 1, -1, -1, dtype=torch.float32, device=log_probs.device)
        return -torch.mean(p_rewards[torch.from_numpy(mask).to(log_probs.device)] * log_probs)

    def generate_episodes(self, list_f, walksat=False):
        losses = []
        all_backflips = []
        all_flips = []
        num_sols = 0
        for f in list_f:
            sat, flips, backflips, log_probs = self.generate_episode_reinforce(f, walksat)
            all_flips.append(flips)
            all_backflips.append(backflips)
            if sat and flips > 0 and not all(map(lambda x: x is None, log_probs)):
                loss = self.reinforce_loss(log_probs)
                losses.append(loss)
            num_sols += sat    
        if losses:
            losses = torch.stack(losses).sum()  
        return all_flips, all_backflips, losses, num_sols/self.max_tries
    
    
    def evaluate(self, data, walksat=False):
        all_flips = []
        all_backflips = []
        mean_losses = []
        accuracy = []
        self.policy.eval()
        for f in data:
            list_f = [f for i in range(self.max_tries)]
            flips, backflips, losses, acc = self.generate_episodes(list_f, walksat)
            all_flips.append(flips)
            all_backflips.append(backflips)
            if losses:
                mean_losses.append(losses.item())
            accuracy.append(acc)
            mean_loss = None
            if mean_losses:
                mean_loss = np.mean(mean_losses)
        all_flips = np.array(all_flips).reshape(-1)
        all_backflips = np.array(all_backflips).reshape(-1)
        return all_flips, all_backflips,  mean_loss, np.mean(accuracy)
        
    def train_epoch(self, optimizer, data):
        losses = []
        all_flips = []
        all_backflips = []
        mean_losses = []
        accuracy = []
        np.random.shuffle(data)
        k = self.max_tries
        batches = [data[i:i+k] for i in range(0, len(data), k)]
        for list_f in batches:
            self.policy.train()
            flips, backflips, loss, acc = self.generate_episodes(list_f)
            if acc > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                mean_losses.append(loss.item())
            all_flips.append(flips)
            all_backflips.append(backflips)
            accuracy.append(acc)
        mean_loss = -1
        if mean_losses:
            mean_loss = np.mean(mean_losses)
        return all_flips, all_backflips, mean_loss, np.mean(accuracy) 


def change_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr



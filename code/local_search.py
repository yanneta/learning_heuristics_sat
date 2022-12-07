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
from torch.distributions.bernoulli import Bernoulli 

class SATLearner:
    def __init__(self, policy, noise_policy, max_flips=10000, p=0.5):
        self.policy = policy
        self.noise_policy = noise_policy
        self.steps_since_improv = 0
        self.max_flips = max_flips
        self.p = p
        self.age = []
        self.age2 = []
        self.last_10 = []
        self.sol = []
        self.flipped = set()
        self.flips = 0
        self.backflipped = 0

    def compute_true_lit_count(self, clauses):
        n_clauses = len(clauses)
        true_lit_count = [0] * n_clauses
        for index in range(n_clauses):
            for literal in clauses[index]:
                if self.sol[abs(literal)] == literal:
                    true_lit_count[index] += 1
        return true_lit_count

    def do_flip(self, literal, occur_list):
        for i in occur_list[literal]:
            self.true_lit_count[i] += 1
        for i in occur_list[-literal]:
            self.true_lit_count[i] -= 1
        self.sol[abs(literal)] *= -1

    def normalize_breaks(self, x):
        #return np.minimum(x, 5)/5
        return x/5

    def normalize_breaks2(self, x):
        x = np.minimum(x, 5)
        return np.log(x + 1)/np.log(6)

    def stats_per_clause(self, f, list_literals):
        """ computes the featutes needed for the model
        """
        variables = [abs(v) for v in list_literals]
        breaks = np.zeros(len(variables))
        last_5 = self.last_10[:5]
        for i, literal in enumerate(list_literals):
            broken_count = 0
            for index in f.occur_list[-literal]:
                if self.true_lit_count[index] == 1:
                    broken_count += 1
            breaks[i] = broken_count
        breaks = self.normalize_breaks2(breaks)
        in_last_10 = np.array([int(v in self.last_10) for v in variables]) 
        age = np.array([self.age[v] for v in variables])/(self.flips)
        age2 = np.array([self.age2[v] for v in variables])/(self.flips)
        in_last_5 = np.array([int(v in last_5) for v in variables])
        x = np.stack([breaks, in_last_10, in_last_5, age, age2], axis=1)
        return x

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

    def select_literal_walksat(self, f, unsat_clause):
        if random.random() < self.p:
            literal = random.choice(unsat_clause)
        else:
            _, literal = self.walksat_step(f, unsat_clause)
        return literal, None

    def sample_estimate_p(self, f):
        num_clauses = len(f.clauses)
        x = self.steps_since_improv/num_clauses 
        x = np.array([x])
        x = torch.from_numpy(x[None,]).float()
        p = self.noise_policy(x)
        m = Bernoulli(p)
        sample = m.sample()
        #print("p", p)
        return sample[0], m.log_prob(x)[0]

    def select_literal(self, f, list_literals):
        log_prob = None
        sample, log_prob_p = self.sample_estimate_p(f)
        #print(sample, log_prob_p)
        if sample == 1:
            literal = random.choice(list_literals)
        else:
            literal, log_prob = self.reinforce_step(f, list_literals)
            v = abs(literal)
            self.age2[v] = self.flips
            self.last_10.insert(0, v)
            self.last_10 = self.last_10[:10]
        return literal, log_prob, log_prob_p

    def update_stats(self, f, literal):
        v = abs(literal)
        if v not in self.flipped:
            self.flipped.add(v)
        else:
            self.backflipped += 1
        self.do_flip(literal, f.occur_list)
        self.age[v] = self.flips

class WalkSATLN(SATLearner):
    def __init__(self, policy, noise_policy, max_tries=10, max_flips=10000, p=0.5, discount=0.5):
        super().__init__(policy, noise_policy,  max_flips, p)
        self.max_tries = max_tries
        self.discount = discount
        self.unsat_clauses = []
        
    def select_variable_reinforce(self, x):
        logit = self.policy(x)
        prob = F.softmax(logit, dim=0)
        dist = Categorical(prob.view(-1))
        v = dist.sample()
        return v, dist.log_prob(v)
    
    def reinforce_step(self, f, list_literals):
        x = self.stats_per_clause(f, list_literals)
        x = torch.from_numpy(x).float()
        index, log_prob = self.select_variable_reinforce(x)
        literal = list_literals[index]
        return literal, log_prob

    def generate_episode_reinforce(self, f, walksat):
        self.sol = [x if random.random() < 0.5 else -x for x in range(f.n_variables + 1)]
        self.true_lit_count = self.compute_true_lit_count(f.clauses)
        self.age = np.zeros(f.n_variables + 1)
        self.age2 = np.zeros(f.n_variables + 1)
        self.flipped = set()
        log_probs = []
        log_probs_p = []
        self.flips = 0
        self.backflipped = 0
        self.last_10 = []
        self.steps_since_improv = 0
        num_unsat_clauses = len(f.clauses)
        while self.flips < self.max_flips:
            unsat_clause_indices = [k for k in range(len(f.clauses)) if self.true_lit_count[k] == 0]
            if num_unsat_clauses > len(unsat_clause_indices):
                self.steps_since_improv = 0
                num_unsat_clauses = len(unsat_clause_indices)
            else:
                self.steps_since_improv += 1
            sat = not unsat_clause_indices
            if sat:
                break
            self.flips += 1
            if walksat:
                unsat_clause = f.clauses[random.choice(unsat_clause_indices)]
                literal, log_prob = self.select_literal_walksat(f, unsat_clause)
                log_prob_p = None
            else:
                indeces = np.random.choice(unsat_clause_indices, 2)
                list_literals = f.clauses[indeces[0]] # +  f.clauses[indeces[1]] 
                literal, log_prob, log_prob_p = self.select_literal(f, list_literals)
            self.update_stats(f, literal)
            log_probs.append(log_prob)
            log_probs_p.append(log_prob_p)

        return sat, self.flips, self.backflipped, log_probs, log_probs_p

    def reinforce_loss(self, log_probs, log_probs_p):
        T = len(log_probs)
        log_probs_filtered = []
        mask = np.zeros(T, dtype=bool)
        for i, x in enumerate(log_probs):
            if x is not None:
                log_probs_filtered.append(x)
                mask[i] = 1

        log_probs = torch.stack(log_probs_filtered)
        p_rewards = self.discount ** torch.arange(T - 1, -1, -1, dtype=torch.float32, device=log_probs.device)
        loss = -torch.mean(p_rewards[torch.from_numpy(mask).to(log_probs.device)] * log_probs)
        loss_p = -torch.mean(p_rewards * torch.stack(log_probs_p))
        return loss, loss_p

    def generate_episodes(self, list_f, walksat=False):
        losses = []
        losses_p = []
        all_backflips = []
        all_flips = []
        sats = []
        for f in list_f:
            sat, flips, backflips, log_probs, log_probs_p = self.generate_episode_reinforce(f, walksat)
            all_flips.append(flips)
            all_backflips.append(backflips)
            if sat and flips > 0 and not all(map(lambda x: x is None, log_probs)):
                loss, loss_p = self.reinforce_loss(log_probs, log_probs_p)
                losses.append(loss)
                losses_p.append(loss_p)
            sats.append(sat)    
        if losses:
            losses = torch.stack(losses).sum()
            losses_p = torch.stack(losses_p).sum()
        return all_flips, all_backflips, losses, losses_p, np.array(sats)
    
    
    def evaluate(self, data, walksat=False):
        all_flips = []
        all_backflips = []
        mean_losses = []
        accuracy = []
        self.policy.eval()
        self.noise_policy.eval()
        for f in data:
            list_f = [f for i in range(self.max_tries)]
            flips, backflips, losses, losses_p, sats = self.generate_episodes(list_f, walksat)
            solved = sats.max()
            all_flips.append(flips)
            all_backflips.append(backflips)
            if losses:
                mean_losses.append(losses.item())
            accuracy.append(solved)
            mean_loss = None
            if mean_losses:
                mean_loss = np.mean(mean_losses)
        all_flips = np.array(all_flips).reshape(-1)
        all_backflips = np.array(all_backflips).reshape(-1)
        return all_flips, all_backflips,  mean_loss, np.mean(accuracy)
        
    def train_epoch(self, optimizer, noise_optimizer, data):
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
            self.noise_policy.train()
            flips, backflips, loss, loss_p, sats = self.generate_episodes(list_f)
            acc = sats.mean()
            if acc > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                mean_losses.append(loss.item())
                noise_optimizer.zero_grad()
                loss_p.backward()
                noise_optimizer.step()

            all_flips.append(flips)
            all_backflips.append(backflips)
            accuracy.append(acc)
        mean_loss = -1
        if mean_losses:
            mean_loss = np.mean(mean_losses)
        return all_flips, all_backflips, mean_loss, np.mean(accuracy) 



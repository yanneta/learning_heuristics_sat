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
from scipy.special import softmax
from utils import *


def flatten(l):
    return [item for sublist in l for item in sublist]

class SATLearner:
    def __init__(self, policy, noise_policy, train_noise=False, max_flips=10000, p=0.5):
        self.policy = policy
        self.model_np =()
        self.noise_policy = noise_policy
        self.steps_since_improv = 0
        self.max_flips = max_flips
        self.p = p
        self.age = []
        self.age2 = []
        self.last_10 = []
        self.sol = []
        self.flips = 0
        self.train_noise = train_noise

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

    def sample_estimate_p(self, f):
        num_clauses = len(f.clauses)
        x = self.steps_since_improv/num_clauses 
        x = np.array([x, x*x])
        x = torch.from_numpy(x[None,]).float()
        p = self.noise_policy(x)
        m = Bernoulli(p)
        sample = m.sample()
        #print(self.flips, p.item(), self.steps_since_improv)
        return sample[0], m.log_prob(sample)[0]

    def select_literal_eval(self, f, list_literals):
        sample = int(random.random() < self.p)
        if sample == 1:
            literal = random.choice(list_literals)
        else:
            literal = self.reinforce_step_np(f, list_literals)
            v = abs(literal)
            self.age2[v] = self.flips
            self.last_10.insert(0, v)
            self.last_10 = self.last_10[:10]
        return literal

    def select_literal(self, f, list_literals):
        log_prob = None
        if self.train_noise:
            sample, log_prob_p = self.sample_estimate_p(f)
        else:
            sample = int(random.random() < self.p)
            log_prob_p = 0
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
        self.do_flip(literal, f.occur_list)
        self.age[v] = self.flips

    def eval_model_np(self, x):
        (w, b) = self.model_np
        return (x*w).sum(axis=1) +b

    def select_variable_reinforce_np(self, x):
        logit = self.eval_model_np(x)
        prob = softmax(logit)
        sample = np.random.multinomial(1, prob, size=1)[0]
        index = sample.nonzero()[0]
        return index[0]

    def reinforce_step_np(self, f, list_literals):
        x = self.stats_per_clause(f, list_literals)
        index = self.select_variable_reinforce_np(x)
        literal = list_literals[index]
        return literal

class WalkSATLN(SATLearner):
    def __init__(self, policy, noise_policy,  train_noise=False, max_tries=10, max_flips=10000, p=0.5, discount=0.5):
        super().__init__(policy, noise_policy, train_noise, max_flips, p)
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


    def init_all(self, f):
        self.sol = [x if random.random() < 0.5 else -x for x in range(f.n_variables + 1)]
        self.true_lit_count = self.compute_true_lit_count(f.clauses)
        self.age = np.zeros(f.n_variables + 1)
        self.age2 = np.zeros(f.n_variables + 1)
        self.flips = 0
        self.last_10 = []
        self.steps_since_improv = 0

    def generate_episode_reinforce_eval(self, f):
        self.init_all(f)
        num_unsat_clauses = len(f.clauses)
        while self.flips < self.max_flips:
            unsat_clause_indices = [k for k in range(len(f.clauses)) if self.true_lit_count[k] == 0]
            sat = not unsat_clause_indices
            if sat:
                break
            self.flips += 1
            indeces = np.random.choice(unsat_clause_indices, 1)
            list_literals = f.clauses[indeces[0]]
            literal = self.select_literal_eval(f, list_literals)
            self.update_stats(f, literal)
        return sat, self.flips

    def generate_episode_reinforce(self, f):
        self.init_all(f)
        log_probs = []
        log_probs_p = []
        num_unsat_clauses = len(f.clauses)
        while self.flips < self.max_flips:
            unsat_clause_indices = [k for k in range(len(f.clauses)) if self.true_lit_count[k] == 0]
            if self.train_noise:
                if num_unsat_clauses > len(unsat_clause_indices):
                    self.steps_since_improv = 0
                    num_unsat_clauses = len(unsat_clause_indices)
                else:
                    self.steps_since_improv += 1
            sat = not unsat_clause_indices
            if sat:
                break
            self.flips += 1
            indeces = np.random.choice(unsat_clause_indices, 1)
            list_literals = f.clauses[indeces[0]] 
            literal, log_prob, log_prob_p = self.select_literal(f, list_literals)
            self.update_stats(f, literal)
            log_probs.append(log_prob)
            if self.train_noise:
                log_probs_p.append(log_prob_p)
        return sat, self.flips, log_probs, log_probs_p

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
        loss_p = 0
        if self.train_noise:
            loss_p = -torch.mean(p_rewards * torch.stack(log_probs_p))
        return loss, loss_p

    def generate_episodes(self, list_f):
        losses = []
        losses_p = []
        all_flips = []
        sats = []
        for f in list_f:
            sat, flips, log_probs, log_probs_p = self.generate_episode_reinforce(f)
            all_flips.append(flips)
            if sat and flips > 0 and not all(map(lambda x: x is None, log_probs)):
                loss, loss_p = self.reinforce_loss(log_probs, log_probs_p)
                losses.append(loss)
                losses_p.append(loss_p)
            sats.append(sat)    
        if losses:
            losses = torch.stack(losses).sum()
            if self.train_noise:
                losses_p = torch.stack(losses_p).sum()
            else:
                losses_p = 0
        return all_flips, losses, losses_p, np.array(sats)
    
    def generate_episodes_eval(self, f):
        all_flips = []
        sats = []
        for i in range(self.max_tries):
            sat, flips = self.generate_episode_reinforce_eval(f)
            all_flips.append(flips)
            sats.append(sat)
        return np.median(all_flips),  np.mean(all_flips), np.array(sats).max()

    def update_p(self):
        self.noise_policy.eval()
        x = np.array([0, 0])
        x = torch.from_numpy(x[None,]).float()
        self.p = self.noise_policy(x)[0][0].item()
        

    def evaluate(self, data):
        med_flips = []
        mean_flips = []
        accuracy = []
        self.policy.eval()
        self.model_np = model_to_numpy(self.policy)
        if self.train_noise:
            self.update_p()
        for i, f in enumerate(data):
            med_f, mean_f, solved = self.generate_episodes_eval(f)
            med_flips.append(med_f)
            mean_flips.append(mean_f)
            accuracy.append(solved)
        return med_flips, mean_flips,  np.mean(accuracy)

    def train_epoch(self, optimizer, noise_optimizer, data):
        losses = []
        all_flips = []
        mean_losses = []
        accuracy = []
        np.random.shuffle(data)
        k = self.max_tries
        batches = [data[i:i+k] for i in range(0, len(data), k)]
        for list_f in batches:
            self.policy.train()
            self.noise_policy.train()
            flips, loss, loss_p, sats = self.generate_episodes(list_f)
            acc = sats.mean()
            if acc > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                mean_losses.append(loss.item())
                if self.train_noise:
                    noise_optimizer.zero_grad()
                    loss_p.backward()
                    noise_optimizer.step()

            all_flips.append(flips)
            accuracy.append(acc)
        mean_loss = -1
        if mean_losses:
            mean_loss = np.mean(mean_losses)
        return flatten(all_flips), mean_loss, np.mean(accuracy) 



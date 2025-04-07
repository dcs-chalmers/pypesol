
import numpy as np
import time

from math import pi, sqrt, comb
from random import sample, shuffle, choices
from itertools import permutations, combinations
from collections import defaultdict
from copy import deepcopy
from scipy.optimize import linear_sum_assignment

from os import listdir, mkdir
from os.path import join, exists

from utils import *
from optimizer import *

def naive_assignment(P, C):
    pass

"""
    Matching version E-energy2020 / Applied Energy:
        -> Only matching (Consumer,Prosumer) Pairs here.
        -> All Consumer-Prosumer pairs are allowed here.
        (ie. one-to-one assignment problem)
"""

# total saving -- data contains (pre)-computed weight (eg. saving) for each possible pair

def match_pairs(data, pairs):
    return sum(data[(h1,h2)] for h1, h2 in pairs)

def match(data, prosumers, pairing):
    return match_pairs(data, zip(prosumers,pairing))

# matching algorithms for pairs

# this is basically the "naive" assignment with ordered prosumers and consumers
def greedy_largest(data, avg_loads, prosumers, consumers):
    pairing = sorted([(avg_loads[c], c) for c in consumers])
    return match(data, prosumers, [c for (avg,c) in pairing[-len(prosumers):]])

def greedy_matching(data, prosumers, consumers):
    available = consumers[:]
    total_saving = 0
    for h1 in prosumers:
        saving, h2 = max(((data[(h1,h2)] if (h1,h2) in data else 0),h2) for h2 in available)
        total_saving += saving
        del available[available.index(h2)]
    return total_saving

def best_matching(data, prosumers, consumers, profit=True):
    cost = np.array([[(data[(p,c)] if (p,c) in data else 0) for c in consumers] for p in prosumers])
    row_ind, col_ind = linear_sum_assignment(cost, maximize=profit)
    pairs = [(i,j) for i,j in zip(row_ind,col_ind) if round(cost[(i,j)],3)]
    return match_pairs(data, [(prosumers[i],consumers[j]) for (i,j) in pairs])

def worst_matching(data, prosumers, consumers):
    return best_matching(data, prosumers, consumers, False)


# matching algorithms for groups

"""
greedy matching is IDENTICAL if only driven by common interest in splitting-strategy
ie fair-split, resource-split (1/2) and proportional-split should produce the same matching!
"""

def greedy_matching_group(costs, participants):
    matched = {}
    for h in participants:
        matched[h] = None

    # we only accept groups where all participants win (they are not allowed to loose money)
    groups = sorted((group for group in costs if all(costs[group][h] > 0 for h in group)),
                    key = lambda group: sum(costs[group][h] for h in group), # sort by total costs
                    reverse = True) # largest gains first

    total_saving = 0
    for group in groups:
        if all(not matched[h] for h in group):
            for h in group:
                matched[h] = group
            total_saving += sum(costs[group][h] for h in group)

    # This should not happenned if groups always benefit
    for h in participants:
        if not matched[h]:
            matched[h] = (h,)

    return total_saving/len(participants), matched

def average_gain(matching):
    return sum(costs[group][h] if group in costs else 0
               for group in set(matching[h] for h in matching) for h in group)/len(matching)
                

"""
    Matching in hypergraphs.
        (ie. one-to-many assignment problem)
"""

def random_matching(opt, k=2, neighbors=None):
    prosumers = opt.get_prosumers()
    if not neighbors:   # if no neighborhood function is provided, all prosumer-consumer pairs are allowed
        neighbors = lambda p: opt.get_consumers()
    matching = defaultdict(list)
    is_matched = defaultdict(bool)
    
    for p in prosumers:
        N = [c for c in neighbors(p) if opt.pv[c] == 0 and not is_matched[c]]
        matching[p] = sample(N,min(k-1,len(N)))
        for c in matching[p]:
            is_matched[c] = True

    return matching

def optimal_pairwise_matching(opt, k, weights):
    prosumers = opt.get_prosumers()
    consumers = opt.get_consumers()
    
    matching = defaultdict(list)
    
    cost = np.array([[(weights[(p,c)] if (p,c) in weights else 0)
                      for c in consumers]
                         for p in prosumers*(k-1) # create (k-1)-copies of the edges
                             ])

    row_ind, col_ind = linear_sum_assignment(cost, maximize=True)

    pairs = [(prosumers[x % len(prosumers)],consumers[y])
             for x,y in zip(row_ind,col_ind) if round(cost[(x,y)],3)]

    for (p,c) in pairs:
        matching[p].append(c)
    
    return matching

def calculate_all_pairwise_weights(opt):
    weights = {}
    for p in opt.get_prosumers():
        for c in opt.get_consumers():
            weights[(p,c)] = (opt.optimize(p)+opt.optimize(c))-opt.optimize_community((p,c))
    return weights

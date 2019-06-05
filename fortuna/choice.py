"""
choice.py
Protocols for chosing bandits

Handles the primary functions
"""

__author__ = "Hannah E. Bruce Macdonald"

import numpy as np
import random

def e_greedy(options,e=1.,reverse=False):
    """ Epsilon-greedy algorithm for selecting sampled bandits

    Parameters
    ----------
    options : list
        list of the bandits samples
    e : float, default = 1
        epsilon value between 0 and 1 that determines how random the chosen bandit is. e=0 fully random, e=1 not random
    reverse : bool, default = False
        reverse selection criteria, i.e. prioritise sampling of minimum results

    Returns
    -------
    int
        index of the selected bandit

    """
    assert (e <= 1. and e >= 0.), f'epsilon in e-greedy algorithm must be between 0 (random) and 1 (greedy).\n epsilon chosen is {e}'

    # with a probability of e, return a random choice to sample
    if np.random.rand(1) > e:
        return random.choice([x for x in range(0,len(options))])

    # otherwise return the maximum sample, or minimum sample if choice is reversed
    if not reverse:
        return np.argmax(options)
    else:
        return np.argmin(options)


def boltzmann(options,T,reverse=False):
    """Short summary.

    Parameters
    ----------
    options : type
        Description of parameter `options`.
    temperature : type
        Description of parameter `temperature`.
    reverse : type
        Description of parameter `reverse`.

    Returns
    -------
    type
        Description of returned object.

    """
    if not reverse:
        expectations = [np.exp(i/T) for i in options]
    else:
        expectations = [np.exp(-i/T) for i in options]
    N = np.sum(expectations)
    normalised = [i/N for i in expectations]
    chosen = np.random.choice(options,p=normalised)
    return options.index(chosen)

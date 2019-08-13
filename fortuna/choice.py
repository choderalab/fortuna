"""
choice.py
Protocols for chosing bandits

Handles the primary functions
"""

__author__ = "Hannah E. Bruce Macdonald"

import numpy as np
import random


def e_greedy(options, e=0., reverse=False):
    """ Epsilon-greedy algorithm for selecting sampled bandits

    Parameters
    ----------
    options : list
        list of the bandits samples
    e : float, default = 0
        epsilon value between 0 and 1 that determines how random the chosen bandit is. e=0 fully greedy, e=1 random
    reverse : bool, default = False
        reverse selection criteria, i.e. prioritise sampling of minimum results

    Returns
    -------
    int
        index of the selected bandit

    """
    assert (e <= 1. and e >= 0.
            ), f'epsilon in e-greedy algorithm must be between 0 (random) and 1 (greedy).\n epsilon chosen is {e}'

    # with a probability of e, return a random choice to sample
    if np.random.rand(1) < e:
        return random.choice([x for x in range(0, len(options))])

    # otherwise return the maximum sample, or minimum sample if choice is reversed
    if not reverse:
        return np.argmax(options)
    else:
        return np.argmin(options)


def epsilon_first(options, stage, reverse=False):
    """Epsilon-first algorithm for sampling bandits. An initial stage of random sampling (stage=0) is performed before a second stage of epislon-greedy sampling

    Parameters
    ----------
    options : list
        list of the bandits samples
    stage : int
        Either 0 or 1 to indicate random or epsilon sampling respectively
    reverse : bool, default = false
        reverse selection criteria, i.e. prioritise sampling of minimum results

    Returns
    -------
    int
        index of the selected bandit

    """
    assert (stage in [
        0, 1
    ]), 'Stage {} not recognised. Only stage 0 (random sampling) or stage 1 (epsilon sampling) are allowed'
    if stage == 0:
        return random.choice([x for x in range(0, len(options))])
    else:
        return epsilon(options, e=0., reverse=reverse)


def epsilon_decreasing(options, iteration, rate=1., reverse=False):
    """Epsilon-greedy algorithm, where the value of epilon decreases with simulation time. This reflects a regime where the sampling protocol is more random at the beginning, and becomes increasingly greedy. The rate of the decay of epsilon can be changed.

    Parameters
    ----------
    options : list
        list of the bandits samples
    iteration : int
        iteration number of the bayesian bandit process
    rate : float, default=1.
        the rate of decay of epsilon.
    reverse : bool, default = false
        reverse selection criteria, i.e. prioritise sampling of minimum results

    Returns
    -------
    int
        index of the selected bandit

    """
    e = np.exp(-steps * rate)
    return epsilon(options, e=e, reverse=reverse)


def boltzmann(options, T=1, reverse=False):
    """ Boltzmann algorithm for selecting sampled bandits

    Parameters
    ----------
    options : list
        list of the bandits samples
    T : float, default = 1
        temperature value between 0 and infinity that determines how random the chosen bandit is. T=0 will behave like e_greedy(e=1) and as T -> inf, the boltzmann selection is increasingly random
    reverse : bool, default = False
        reverse selection criteria, i.e. prioritise sampling of minimum results

    Returns
    -------
    int
        index of the selected bandit

    """
    if not reverse:
        expectations = [np.exp(i / T) for i in options]
    else:
        expectations = [np.exp(-i / T) for i in options]
    N = np.sum(expectations)
    normalised = [i / N for i in expectations]
    chosen = np.random.choice(options, p=normalised)
    return options.index(chosen)


def UCB(options, steps, iteration):
    """ Upper-confidence bounds (UCB) algorithm, which models optimisim in the face of uncertainty. i.e. sampling of less sampled bandits is increased

    Parameters
    ----------
    options : list
        list of the bandits samples
    steps : list
        list of the number of times each bandit has been sampled
    iteration : int
        the iteration of the adaptive sampling scheme

    Returns
    -------
    int
        index of the selected bandit

    """

    assert (0 not in steps), 'UCB algorithm requires an initial single pull of each bandit'
    weights = [(2 * np.log(iteration) / n)**0.5 for n in steps]
    adjusted_options = [(x + y) for x, y in zip(options, weights)]
    return np.argmax(adjusted_options)

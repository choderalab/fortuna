"""
Unit and regression test for the fortuna package.
"""

# Import package, test suite, and other packages as needed
import fortuna
import pytest
import sys

def test_fortuna_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "fortuna" in sys.modules

def test_bernoulli(probability=0.7,tolerance=0.1,nsteps=500):
    assert (probability >=0. and probability <= 1.), f'Testing probability must be between 0 and 1. Probability = {probability}'
    bandit = fortuna.bandits.Bernoulli(probability)

    total = 0
    for i in range(0,nsteps):
        reward = bandit.pull()
        bandit.update(reward)
        total += reward

    average = float(total) / float(nsteps)
    # check that the result of pulling the bandit follows the probability that it should
    assert ( abs(average - probability) <= tolerance ), f'The average reward is not close enough to the liklihood. Probability = {probability}, average = {average} with tolerance = {tolerance}'

    # check that the posterior is in good agreement with the probability
    total = 0
    for i in range(0,nsteps):
        sample = bandit.sample()
        total += sample

    average = float(total) / float(nsteps)
    assert ( abs(average - probability) <= tolerance ), f'The average of the posterior is not close enough to the likelihood. Probability = {probability}, average = {average} with tolerance = {tolerance}'

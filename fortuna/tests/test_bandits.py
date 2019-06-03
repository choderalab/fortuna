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
    assert ( abs(average - probability) <= tolerance ), f'The average reward is not close enough to the Bernoulli probability. Probability = {probability}, average = {average} with tolerance = {tolerance}'

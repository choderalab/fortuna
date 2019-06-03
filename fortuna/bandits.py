"""
bayesianbandits.py
Methodologies for adaptive sampling

Handles the primary functions
"""

__author__ = "Hannah E. Bruce Macdonald"

import numpy as np
from scipy.stats import bernoulli


class Bernoulli(object):
    """ Bernoulli-beta bayesian bandit.

    Likelihood follows a Bernoulli distribution, of probability p.
    The conjugate prior is a Beta function with hyperparameters a, b.

    Parameters
    ----------

    p is the


    """
    def __init__(self,probability):
        self.p = probability
        assert (self.p >= 0. and self.p <=1.), 'probability distribution must fall between 0. and 1.'
        self.a = 0
        self.b = 0
        self.steps = 0

    def assign_prior(self,a,b):
        """ Assign an alternative informed prior other than the default a=0,b=0

        Parameters
        ----------
        a : int
            value of alpha for Beta function
        b : type
            value of beta for Beta function

        Returns
        -------

        """
        assert (self.steps == 0), 'Prior can only be assigned before sampling of the bandit'
        self.a = a
        self.b = b

    def sample(self):
        """Sample the bandit prior distribution

        Parameters
        ----------


        Returns
        -------
        float
            random variable from the distribution Beta(a,b)

        """
        return np.random.beta(self.a,self.b)

    def pull(self):
        """'Pull' the bandit, which involves sampling from Bernoulli(p).
        For adaptive sampling, this class should be subclassed, and pull() replaced
        with a function tailored to the particular application.

        Parameters
        ----------


        Returns
        -------
        type
            Description of returned object.

        """
        return bernoulli.rvs(self.p)

    def update(self,reward):
        """Update the prior hyperparameters

        Parameters
        ----------
        reward : bool
            True if reward is true, false otherwise

        Returns
        -------

        """
        # update the number of steps
        self.steps +=1
        self.a += reward
        self.b += (1 - reward)

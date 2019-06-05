"""

http://tokic.com/www/tokicm/publikationen/papers/AdaptiveEpsilonGreedyExploration.pdf
http://www.tokic.com/www/tokicm/publikationen/papers/KI2011.pdf

"""

from fortuna.bandits import Bernoulli
import copy

class VDBEBernoulli(Bernoulli):
    def __init__(self,p):
        super(VDBEBernoulli,self).__init__(p)
        self.Q = 0
        self.Q_t = 0
        self.TDerror = 0
        self.epsilon = 1
        self.a = 1./(1.+self.steps)

    def update_Q(self,reward):
        self.Q = copy.deepcopy(self.Q_t)
        self.Q_t = self.Q + self.a*(reward-self.Q)
        self.TDerror = np.abs(self.Q - self.Q_t)

# where delta = len(bandits)**-1
def VDBE_epsilon(bandit,delta):

    exponent =
    function = (1 - exponent)/(1 + exponent)
    epsilon = delta*function+(1-delta)*bandit.epsilon
    return epsilon
function = []

from cuqipy_pytorch.distribution import Gaussian, Uniform
from cuqi.distribution import JointDistribution
from cuqipy_pytorch.sampler import NUTS

import torch as xp
import numpy as np
import matplotlib.pyplot as plt

from tools import U_class


# This class assembles the KL expansion as the forward model
class regression():
    def __init__(self, N=256):
        self.N = N
        self.field = U_class(N)

    def forward(self, v, s):
        u = self.field.make_u_from_s_v(s, v)
        return u

    def forward_np(self, v, s):
        u = self.field.make_u_from_s_v(s, v)
        return u.detach().numpy()

def run_NUTS():
    # loading observation file
    obs_data = np.load('./obs/obs_power_1024_model_selection.npz')
    y_true = obs_data['y_true'] # true signal
    noise_vec = obs_data['noise_vec1'] # noise vector
    # noise_vec = obs_data['noise_vec2'] # uncomment for the second noise vector
    N = int(obs_data['N'])
    sigma_noise = 0.1 # noise level 10%
    sigma2 = sigma_noise**2
    noise = noise_vec*np.linalg.norm(y_true)*sigma_noise
    y_obs = xp.asarray(y_true + noise) # noisy measurement

    # defining the forward problem
    problem = regression(N)

    # Helper Idendity matrix
    I = xp.ones(N)

    # Lets build the Bayesian generative model for V, S, V
    s = Uniform(0,10)
    v = Gaussian(xp.zeros(N), 1)
    y = Gaussian(problem.forward, sigma_noise**2*I)

    # Joint
    J = JointDistribution(s,v,y)

    # Sampler
    samples = NUTS(J(y=y_obs)).sample(20000, 5000)

    # saving samples
    np.savez( './stats/power_1024_model_selection_1.npz', s=samples['s'].samples, v=samples['v'].samples)


if __name__ == '__main__':
    run_NUTS()
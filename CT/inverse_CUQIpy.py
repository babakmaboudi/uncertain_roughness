#import sys;
#sys.path.append("../cuqipy")

import numpy as np
import torch as xp
import matplotlib.pyplot as plt
from inclusion import U_class, image, full_tensor_sino

import arviz
from cuqi.distribution import Gaussian, Uniform, JointDistribution
from cuqi.sampler import Gibbs, MH, pCN

class forward_map():
    def __init__(self,N,view_angles,s=2):
        self.U = U_class(N=N)

        self.radon = full_tensor_sino()
        self.view_angles = xp.asarray(view_angles)

    def forward(self,s,v):
        if isinstance(s, np.ndarray):
            s = xp.asarray(s)
        if isinstance(v, np.ndarray):
            v = xp.asarray(v)
        self.U.compute_sqrt_lambda(s)
        u = self.U.make_u_from_v( v )
        sino = self.radon.make_sino(u,self.view_angles)
        return sino.numpy().flatten()

def F(s,v):
    if isinstance(s, np.ndarray):
        s = xp.asarray(s)
    if isinstance(v, np.ndarray):
        v = xp.asarray(v)
    out = forward.forward(s,v)
    if isinstance(out, xp.Tensor):
        out = out.numpy()
    return out.flatten()

class Metro(MH):
    def step(self, x):
        self.x0 = x
        self.scale = 0.8
        return self.sample(20, ).samples[:,-1]

    def _print_progress(*args, **kwargs):
        pass

class PCN(pCN):
    def step(self, x):
        self.x0 = x
        self.scale = 0.07
        return self.sample(20).samples[:,-1]

    def _print_progress(*args, **kwargs):
        pass

def run_gibbs():
    # loading observation file
    obs_data = np.load('./obs/obs_gear_90.npz')
    N = obs_data['N'] # discretization size
    sino = xp.asarray( obs_data['sino'] )
    noise_vec = xp.asarray( obs_data['noise_vec'] ) # noise vector
    view_angles = obs_data['view_angles'] # view angles

    # defining the forward problem
    forward = forward_map(N,view_angles)

    sigma = np.linalg.norm(sino)/100
    sigma2 = sigma*sigma
    y_obs = sino + sigma*noise_vec
    y_obs_flat = y_obs.flatten()

    log_like = lambda v,s: - ( 0.5*np.sum(  (forward.forward(s,v) - y_obs)**2)/sigma2 )

    m = len(y_obs_flat)
    Im = np.ones(m)

    # Bayesian model
    s = Uniform(0.5,5)
    v = Gaussian(np.zeros(N) , 1)
    y = Gaussian(forward.forward, sigma2*Im)

    # joint distribution
    P = JointDistribution(s,v,y)

    # Gibbs sampler
    sampler = Gibbs(P(y=y_obs_flat), {'s':Metro, 'v':PCN})

    # run sampler
    samples = sampler.sample(10000)

    np.savez('./stats/stat_gear.npz',samp1=samples['v'].samples.T,samp2=samples['s'].samples.T)

if __name__ == '__main__':
    run_gibbs()



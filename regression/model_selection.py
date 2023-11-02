import numpy as np
import torch as xp
import matplotlib.pyplot as plt

from tools import U_class

import scipy.optimize as opt
from progressbar import progressbar

# this class assembles the KL expansion and also provides the Jacobian information
class regression():
    def __init__(self, N=256):
        self.N = N
        self.field = U_class(N)

    def fix_s(self, s):
        self.field.compute_sqrt_lambda(s)

    def forward(self, v):
        u = self.field.make_u_from_v(v)
        return u

    def jacobian(self, v):
        return self.field.jacobian(v)

    def forward_np(self, v, s):
        u = self.field.make_u_from_v(s, v)
        return u.detach().numpy()

# function for computing the MAP estimate
def compute_MAP(s, y_obs, sigma2):
    N = len(y_obs)
    problem = regression(N)
    problem.fix_s(s)

    v = np.random.standard_normal(N)

    # the negative log posterior in numpy
    functional = lambda v: xp.norm( problem.forward(v) - y_obs, 2 )**2 + sigma2*xp.norm(v, 2)**2

    # the negative log posterior in pytorch
    functional_np = lambda v: functional( xp.asarray(v) ).detach().numpy()

    # defining the Jacobian using the autograd functionality of pytorch
    jac_np = lambda v: xp.autograd.functional.jacobian( functional, xp.asarray(v) ).detach().numpy()


    # initial guess for the optimization
    v0 = np.zeros(N)

    # performing the optimization
    res = opt.minimize(functional_np, v0, jac=jac_np, method='CG')
    v_MAP = res['x']

    return v_MAP

# function for computing the evidence for a given s
def compute_evidence(v_MAP, s, y_obs, sigma2):
    N = len(v_MAP)

    problem = regression(N)
    problem.fix_s(s)

    # defining the negative log posterior
    neg_log_post = lambda v: 0.5*xp.norm( problem.forward(v) - y_obs, 2 )**2/sigma2 + 0.5*xp.norm(v, 2)**2

    # defining the hessian using the autograd functionality of pytorch
    hessian = xp.autograd.functional.hessian( neg_log_post, xp.asarray(v_MAP) )
    d = xp.diag(hessian) # hessian diagonal elements
    
    # computing the terms in the evidence
    term1 = -0.5 * xp.norm( problem.forward(xp.asarray(v_MAP) ) - xp.asarray(y_obs), 2 )**2/sigma2 - 0.5*xp.norm( xp.asarray(v_MAP), 2 )**2
    term2 = - 0.5* xp.sum( xp.log(d) )

    # the final value of the log of the evidence
    log_evidence = term1 + term2
    return log_evidence

def make_evidence_plot():
    # loading the observation data
    obs_data = np.load('./obs/obs_power_1024_model_selection.npz')
    y_true = obs_data['y_true']
    noise_vec1 = obs_data['noise_vec1']
    N = int(obs_data['N'])
    sigma_noise = 0.1
    sigma2 = sigma_noise**2
    noise = noise_vec1*np.linalg.norm(y_true)*sigma_noise
    y_obs = xp.asarray(y_true + noise)

    # discretizing s to compute the evidence
    s_values = np.linspace(0.75,2,30)
    evidence = []
    for s in progressbar( s_values ):
        v_MAP = compute_MAP(s, y_obs, sigma2) # Compute the MAP for a given s
        e = compute_evidence( v_MAP, s, y_obs, sigma2 ) # compute the evidence for s
        evidence.append(e)

    idx = np.argmax(evidence) # Finding the with the highest evidence
    print(s_values[idx])

    f,ax = plt.subplots(1, figsize=(6.4, 2.4))

    ax.plot(s_values,evidence, color='blue', label = 'y1', linewidth=2.)

    # repeating for another noise vector
    noise_vec2 = obs_data['noise_vec2']
    sigma_noise = 0.1
    sigma2 = sigma_noise**2
    noise = noise_vec2*np.linalg.norm(y_true)*sigma_noise
    y_obs = xp.asarray(y_true + noise)

    s_values = np.linspace(0.75,2,30)
    evidence = []
    for s in progressbar( s_values ):
        v_MAP = compute_MAP(s, y_obs, sigma2)
        e = compute_evidence( v_MAP, s, y_obs, sigma2 )
        evidence.append(e)

    idx = np.argmax(evidence)
    print(s_values[idx])

    ax.plot(s_values,evidence, color = 'red', label='y2', linewidth=2.)

    ax.set_ylabel('log evidence', fontsize=18)
    ax.set_xlabel( 's', fontsize=18 )
    plt.tight_layout()
    ax.grid(axis='y')

    ax.legend(prop={'size': 16})
    plt.show()

if __name__ == '__main__':
    make_evidence_plot()
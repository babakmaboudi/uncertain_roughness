import numpy as np
import torch as xp
import matplotlib.pyplot as plt
from inclusion import U_class, image, full_tensor_sino
import inclusion_gear

import arviz

def plot_uq_starshaped():
    # setting the angle of the test problem
    angle = 90

    f, ax = plt.subplots(1)

    # loading the observation and true inclusion
    obs_data = np.load('./obs/obs_la_{}.npz'.format(angle))
    N = obs_data['N']
    if angle == 60:
        N=512
    s_true = obs_data['s']
    v_true = xp.asarray( obs_data['v'] )
    u_true = xp.asarray( obs_data['u'] )
    sino = xp.asarray( obs_data['sino'] )
    noise_vec = xp.asarray( obs_data['noise_vec'] )
    view_angles = obs_data['view_angles']

    # loading the samples
    stats_data = np.load('./stats/stat_la_{}.npz'.format(angle))
    samp_v = stats_data['samp1']
    samp_s = stats_data['samp2']

    # computing the 99% highest posterior density interval (HDI)
    intervals = arviz.hdi( samp_s.reshape(-1), multimodal=True , hdi_prob=0.99)
    U = U_class(N=N)
    image_generator = image()
    image_generator_gear = inclusion_gear.image()

    # computing the mean inclusion
    samp_u = []
    U.compute_sqrt_lambda( np.mean(samp_s) )
    for i in range( samp_v.shape[0] ):
        samp_u.append( (U.make_u_from_v( samp_v[i] ))[None,:] )
    samp_u = xp.cat(samp_u,0)
    print('mean s: ',np.mean(samp_s))

    # computing the posterior mean in the image domain
    u_est = U.make_u_from_s_v( np.mean(samp_s), np.mean(samp_v,axis=0) )

    # ploting the uncertainty band
    image_generator.plot_uq(samp_u, ax, color='skyblue', label=r'99\% HDI')
    # plotting the true boundary
    image_generator.plot_boundary(u_true, ax, color='red', label=r'true inclusion')
    # plotting the posterior mean
    image_generator.plot_boundary( u_est ,ax , color = 'mediumslateblue', label=r'est. inclusion' )

    # plotting the view angles
    va = view_angles[::10]
    image_generator.plot_view_points(ax, va)
    ax.legend()

    plt.show()

def plot_uq_gear():
    # setting the angle of the test problem
    angle = 90

    f, ax = plt.subplots(1)

    # loading the observation and true inclusion
    obs_data = np.load('./obs/obs_gear_{}.npz'.format(angle))
    N = 512 
    u_true = xp.asarray( obs_data['u'] )
    sino = xp.asarray( obs_data['sino'] )
    noise_vec = xp.asarray( obs_data['noise_vec'] )
    view_angles = obs_data['view_angles']

    # loading the samples
    stats_data = np.load('./stats/stat_gear_{}.npz'.format(angle))
    samp_v = stats_data['samp1']
    samp_s = stats_data['samp2']

    # computing the 99% highest posterior density interval (HDI)
    intervals = arviz.hdi( samp_s.reshape(-1), multimodal=True , hdi_prob=0.99)
    U = U_class(N=N)
    image_generator = image()
    image_generator_gear = inclusion_gear.image()

    # computing the mean inclusion
    samp_u = []
    U.compute_sqrt_lambda( np.mean(samp_s) )
    for i in range( samp_v.shape[0] ):
        samp_u.append( (U.make_u_from_v( samp_v[i] ))[None,:] )
    samp_u = xp.cat(samp_u,0)
    print('mean s: ',np.mean(samp_s))

    # computing the posterior mean in the image domain
    u_est = U.make_u_from_s_v( np.mean(samp_s), np.mean(samp_v,axis=0) )

    # ploting the uncertainty band
    image_generator.plot_uq(samp_u, ax, color='skyblue', label=r'99\% HDI')
    # plotting the true boundary
    image_generator_gear.plot_boundary( u_true ,ax, color='r', label='true inclusion' )
    # plotting the posterior mean
    image_generator.plot_boundary( u_est ,ax , color = 'mediumslateblue', label=r'est. inclusion' )

    # plotting the view angles
    va = view_angles[::10]
    image_generator.plot_view_points(ax, va)
    ax.legend()

    plt.show()

if __name__ == '__main__':
    plot_uq_starshaped()
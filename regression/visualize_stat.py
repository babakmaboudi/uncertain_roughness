import numpy as np
import matplotlib.pyplot as plt

import arviz
import seaborn as sns

def plot_uq_s_compare(N, idx):
    obs_data = np.load('./obs/obs_power_{}_model_selection.npz'.format(N) )
    y_true = obs_data['y_true']
    noise_vec = obs_data['noise_vec{}'.format(idx)]
    N = int(obs_data['N'])
    sigma_noise = 0.1
    noise = noise_vec*np.linalg.norm(y_true)*sigma_noise

    stat_path = './stats/power_{}_model_selection_{}.npz'.format(N,idx)
    stat_data = np.load(stat_path)

    print('posterior mean for in case number {}:'.format(idx))
    print(np.mean( stat_data['s'] ) )

    return stat_data['s'].flatten()

def plot_compare_model_selection_s():
    f, ax = plt.subplots( 1 , figsize=(6.4, 2.4))
    N = 1024
    noise_level = 1

    s_samples = []
    # collecting samples for the two cases of noise realization
    for i in range( 2 ):
        s_samples.append( plot_uq_s_compare(N, i+1) )

    s_samples = np.array(s_samples).flatten()

    # labeling each sample
    labels = np.array([ np.ones(20000), 2*np.ones(20000) ]).flatten()

    # collecting the samples in a violin plot using seaborn package
    sns.violinplot(x = labels, y=s_samples, color="skyblue")

    ax.set_ylabel('s', fontsize=18)
    ax.set_xticks([0,1], [ 'y1','y2' ])
    plt.tight_layout()
    ax.grid(axis='y')

    plt.show()

if __name__ == '__main__':
    plot_compare_model_selection_s()
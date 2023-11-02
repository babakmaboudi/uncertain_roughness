import torch as xp
import numpy as np
import arviz

import matplotlib.pyplot as plt
from inclusion import full_tensor_sino

import time

# This class is used to visualize the gear phantom
class image():
    def __init__(self, num_pixel=128, scale=0.05, min_radius=0.2):
        self.num_pixel = num_pixel
        self.scale = scale
        self.min_radius = min_radius

    def plot_boundary(self, v, ax, c=np.zeros(2), label=None, color=None):
        r = np.zeros( len(v) + 1 )
        r[:-1] = v
        r[-1] = v[0]
        theta = np.linspace( 0,2*np.pi, len(v)+1, endpoint=True )

        ax.plot(r*np.cos(theta),r*np.sin(theta),label=label, color=color)
        ax.set_aspect('equal')
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])

    def plot_inclusion(self, v, ax, c=np.zeros(2), label=None):
        r = np.zeros( len(v) + 1 )
        r[:-1] = v
        r[-1] = v[0]
        theta = np.linspace( 0,2*np.pi, len(v)+1, endpoint=True )

        ax.fill(r*np.cos(theta),r*np.sin(theta),label=label, color='r')
        ax.set_aspect('equal')
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])

    def plot_view_points(self,ax,theta):
        for t in theta:
            x = 0.95*np.cos( t )
            y = 0.95*np.sin( t )
            #ax.plot(x,y,'o',color='fuchsia')
            ax.quiver(x,y, -0.15*x, -0.15*y, color='k', angles='xy', scale_units='xy', scale=1)

# this function creates the gear phantom and performs the radon transform and adds additive noise 
def save_gear():
    N = 512
    t = np.linspace(0,2*np.pi,N, endpoint=False)

    v = 0.3*(1 + 0.1*np.tanh(10* np.sin( 10*t ) ))

    radon = full_tensor_sino()
    view_angles = np.linspace(0,np.pi/2,3*64,endpoint=False)
    sino = radon.make_sino(xp.asarray(v),xp.asarray(view_angles) )

    f, axes = plt.subplots(1,3)

    im = image()
    im.plot_boundary(v,axes[0])
    im.plot_inclusion(v,axes[1])
    im.plot_view_points(axes[1],view_angles)
    axes[2].imshow( sino.detach().numpy().T )

    plt.show()

    noise_vec = np.random.standard_normal(sino.shape)
    noise_vec /= np.linalg.norm(noise_vec)

    np.savez( './obs/obs_gear.npz', N=N, u=v, view_angles=view_angles, sino=sino.detach().numpy(), noise_vec=noise_vec )

if __name__ == '__main__':
    save_gear()

import torch as xp
import numpy as np
import arviz

import matplotlib.pyplot as plt

import time

# This class assembles the KL expansion using the FFT
class U_class():
    def __init__(self, N=128):
        self.N = N # discretization size
        mode = xp.asarray(np.concatenate( [ xp.arange(0, int(N/2) ), xp.arange( -int(N/2),0 ) ] )) # Fourier modes associated with the Laplacian operator
        delta = 10. # 1/l where l is the correlation length of the Matern
        delta2 = delta*delta 
        self.delta2pmode2 = delta2 + mode**2 # KL expansion decay coefficients with s=1

    # This will apply s to the expansion coefficients
    def compute_sqrt_lambda(self, s):
        sqrt_lambda = xp.float_power( self.delta2pmode2 , -(s+0.5) )
        norm_factor = xp.linalg.norm(sqrt_lambda)
        self.sqrt_lambda_div = sqrt_lambda/norm_factor

    # Applying the IFFT to v to obtain u
    def make_u_from_v(self, v):
        value = xp.fft.ifft( self.sqrt_lambda_div*v )*float(self.N)
        u =  value.real + value.imag
        return u

    # This will apply s and IFFT at the same time
    def make_u_from_s_v(self, s, v):
        sqrt_lambda = xp.float_power( self.delta2pmode2 , -(s+0.5) )
        norm_factor = xp.linalg.norm(sqrt_lambda)
        sqrt_lambda_div = sqrt_lambda/norm_factor
        value = xp.fft.ifft( sqrt_lambda_div*v )*float(self.N)
        u =  value.real + value.imag
        return u

# This class is used to visualize inclusions
class image():
    def __init__(self, num_pixel=128, scale=0.05, min_radius=0.2):
        self.num_pixel = num_pixel
        self.scale = scale
        self.min_radius = min_radius

    def plot_boundary(self, u, ax, c=np.zeros(2), label=None, color=None):
        r = np.zeros( len(u) + 1 )
        r[:-1] = self.scale * np.exp( u ) + self.min_radius
        r[-1] = r[0]
        theta = np.linspace( 0,2*np.pi, len(u)+1, endpoint=True )

        ax.plot(r*np.cos(theta),r*np.sin(theta),label=label, color=color)
        ax.set_aspect('equal')
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])

    def plot_inclusion(self, u, ax, c=np.zeros(2), label=None):
        r = np.zeros( len(u) + 1 )
        r[:-1] = self.scale * np.exp( u ) + self.min_radius
        r[-1] = r[0]
        theta = np.linspace( 0,2*np.pi, len(u)+1, endpoint=True )

        ax.fill(r*np.cos(theta),r*np.sin(theta),label=label, color='r')
        ax.set_aspect('equal')
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])

    def plot_uq(self, samps_u, ax, c=np.zeros(2), label=None, color=None):
        r = self.scale * np.exp( samps_u ) + self.min_radius
        r = np.concatenate( [r, r[:,0].reshape(-1,1)], axis=1 )
        theta = np.linspace( 0,2*np.pi, samps_u.shape[1]+1, endpoint=True )

        hdi_intervals = []
        for i in range(r.shape[1]):
            local_interval = arviz.hdi( r[:,i], hdi_prob=.99 )
            hdi_intervals.append( local_interval.reshape(-1) )
        hdi_intervals = np.array(hdi_intervals)

        x = hdi_intervals[:,1]*np.cos(theta)
        y = hdi_intervals[:,1]*np.sin(theta)
        ax.fill(x,y,color=color, label=label)
        x = hdi_intervals[:,0]*np.cos(theta)
        y = hdi_intervals[:,0]*np.sin(theta)
        ax.fill(x,y,'w')

    def plot_view_points(self,ax,theta):

        for t in theta:
            x = 0.95*np.cos( t )
            y = 0.95*np.sin( t )
            #ax.plot(x,y,'o',color='fuchsia')
            ax.quiver(x,y, -0.15*x, -0.15*y, color='k', angles='xy', scale_units='xy', scale=1)

# This is the class applies the radon transform to a star-shaped inclusion
class full_tensor_sino():
    def __init__(self, num_pixel=128, scale=0.05, min_radius=0.2):
        self.num_pixel = num_pixel # number of sensors
        self.scale = scale # scale of the star-shaped radius
        self.min_radius = min_radius # minimum radius for the star-shaped inclusion
        self.pixel_location = xp.asarray( np.linspace( -1,1,self.num_pixel ) ) # location of the pixels

    def make_sino(self, u, view_angles):
        r = xp.zeros( len(u) + 1 )
        r[:-1] = self.scale * xp.exp( u ) + self.min_radius
        r[-1] = r[0]
        theta = xp.asarray( np.linspace( 0,2*np.pi, len(u)+1, endpoint=True ) )

        THETA = xp.broadcast_to(theta, (view_angles.shape[0],theta.shape[0]))
        ROT = xp.broadcast_to(view_angles, (theta.shape[0],view_angles.shape[0])).T
        ROTATED_THETA = THETA - ROT
        R = xp.broadcast_to(r, (view_angles.shape[0],r.shape[0]))

        X = R*xp.cos( ROTATED_THETA )
        Y = R*xp.sin( ROTATED_THETA )

        w = self.pixel_location
        XX = xp.broadcast_to(X, (w.shape[0],X.shape[0],X.shape[1])).permute(1,0,2)
        WW = xp.broadcast_to(w, (X.shape[0],X.shape[1],w.shape[0])).permute(0,2,1)

        x_rel_to_line = XX-WW
        crossing_indicator = x_rel_to_line[:,:,:-1]*x_rel_to_line[:,:,1:]

        Xd = xp.diff(X, dim=1)
        Yd = xp.diff(Y, dim=1)
        Xs = xp.roll(X, shifts=-1, dims=1)
        Ys = xp.roll(Y, shifts=-1, dims=1)

        WW2 = xp.broadcast_to(w, (Yd.shape[0],Yd.shape[1],w.shape[0])).permute(0,2,1)
        YYd = xp.broadcast_to(Yd, (w.shape[0],Yd.shape[0],Yd.shape[1])).permute(1,0,2)

        Y_collision = (YYd*WW2 + (Y*Xs)[:,None,:-1] - (Ys*X)[:,None,:-1]) / Xd[:,None,:]

        Y_collision[crossing_indicator>=0] = 0
        Y_collision_sorted = xp.sort(Y_collision, axis=2)[0]

        return xp.sum(xp.diff(Y_collision_sorted, axis=2)[:,:, ::2], axis=2)


        

def compare_sinos():
    N = 256
    U = U_class(N=N)

    s = 2
    v = np.random.standard_normal(N)
    
    U.compute_sqrt_lambda(s)
    u = U.make_u_from_v( xp.asarray(v) )

    radon_vector = vector_sino()
    radon_tensor = tensor_sino()
    radon_full_tensor = full_tensor_sino()
    view_angles = np.linspace(0,np.pi,128,endpoint=False)

    t1 = time.time()
    sino_vector = radon_vector.make_sino(u.detach().numpy(),view_angles)
    print(time.time()-t1)
    t1 = time.time()
    sino_tensor = radon_tensor.make_sino(u,view_angles)
    print(time.time()-t1)
    t1 = time.time()
    sino_full_tensor = radon_full_tensor.make_sino(u,xp.asarray(view_angles) )
    print(time.time()-t1)

    f,axes = plt.subplots(1,3)
    axes[0].imshow(sino_vector)
    axes[1].imshow(sino_tensor.detach().numpy())
    axes[2].imshow(sino_full_tensor.detach().numpy())
    print(np.max(abs(sino_vector-sino_full_tensor.detach().numpy())))
    plt.show()

# this function creates a random star-shaped and performs the radon transform and adds additive noise 
def save_obs():
    N = 256
    U = U_class(N=N)

    s = np.random.uniform(1,2)
    print(s)
    v = np.random.standard_normal(N)
    
    U.compute_sqrt_lambda(s)
    u = U.make_u_from_v( xp.asarray(v) )

    f,ax = plt.subplots(1,2)

    im = image()
    im.plot_boundary(u.detach().numpy(),ax[0])

    radon = full_tensor_sino()
    view_angles = np.linspace(0,np.pi/6,128,endpoint=False)

    sino = radon.make_sino(u,xp.asarray(view_angles) )
    ax[1].imshow(np.rot90(sino))
    plt.show()

    noise_vec = np.random.standard_normal(sino.shape)
    noise_vec /= np.linalg.norm(noise_vec)

    np.savez( './obs/obs_starshaped.npz', N=N, s=s, v=v, u=u.detach().numpy(), view_angles=view_angles, sino=sino.detach().numpy(), noise_vec=noise_vec )

if __name__ == '__main__':
    save_obs()


import torch as xp
import numpy as np

class U_class():
    def __init__(self, N=128):
        self.N = N
        mode = xp.asarray(np.concatenate( [ xp.arange(0, int(N/2) ), xp.arange( -int(N/2),0 ) ] ))
        delta = 10.
        delta2 = delta*delta
        self.delta2pmode2 = delta2 + mode**2

    def compute_sqrt_lambda(self, s):
        sqrt_lambda = xp.float_power( self.delta2pmode2 , -(s+0.5) )
        norm_factor = xp.linalg.norm(sqrt_lambda)
        self.sqrt_lambda_div = sqrt_lambda/norm_factor

    def make_u_from_v(self, v):
        value = xp.fft.ifft( self.sqrt_lambda_div*v )*float(self.N)
        u =  value.real + value.imag
        return u

    def make_u_from_s_v(self, s, v):
        sqrt_lambda = xp.float_power( self.delta2pmode2 , -(s+0.5) )
        norm_factor = xp.linalg.norm(sqrt_lambda)
        sqrt_lambda_div = sqrt_lambda/norm_factor
        value = xp.fft.ifft( sqrt_lambda_div*v )*float(self.N)
        u =  value.real + value.imag
        return u

    def jacobian(self, v):
        return xp.autograd.functional.jacobian( self.make_u_from_v, v )


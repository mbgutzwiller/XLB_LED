from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.equilibrium import Equilibrium_LED
from xlb.operator.operator import Operator
import warp as wp

from typing import Any


def initialize_f_U_num_LED(f, grid, precision_policy, compute_backend):
    if f is None:
        U_0 = grid.create_field(cardinality=20, fill_value=0.0, dtype=precision_policy.compute_precision)
    
    equilibrium = Equilibrium_LED()

    if compute_backend == ComputeBackend.JAX:
        f = equilibrium(U_0)  # TODO

    elif compute_backend == ComputeBackend.WARP:
        U_0 = grid.create_field(cardinality=5, fill_value=0.0, dtype=precision_policy.compute_precision)
        # U_1 = grid.create_field(cardinality=5, fill_value=0.0, dtype=precision_policy.compute_precision)
        u_num_displ_0 = grid.create_field(cardinality=5, fill_value=0.0, dtype=precision_policy.compute_precision)
        u_num_displ_1 = grid.create_field(cardinality=5, fill_value=0.0, dtype=precision_policy.compute_precision)
        f = equilibrium(U_0, f)
    return f, U_0, u_num_displ_0, u_num_displ_1

class Initializer_LED(Operator):
    def __init__(self, velocity_set=None, precision_policy=None, compute_backend=None):
        super().__init__(velocity_set, precision_policy, compute_backend)
    
    def _construct_warp(self):
        _u_num_displ_vector_vec = wp.vec(2, dtype=self.compute_dtype)
        _U_vector_vec = wp.vec(5, dtype=self.compute_dtype)
        _f_vector_vec = wp.vec(20, dtype=self.compute_dtype)

        # Analytical functions to set initial f correctly (non trivial in contrast to fluid LBM)
        @wp.func
        def u_num_displ_func(x: wp.float32, y: wp.float32, t: wp.float32):
            _u_num_displ = _u_num_displ_vector_vec(0.)
            _u_num_displ[0] = wp.exp(-((x-1./2.)**2. + (y-1./2.)**2.) / wp.S_pulse)
            _u_num_displ[1] = wp.exp(-((x-1./2.)**2. + (y-1./2.)**2.) / wp.S_pulse)
            # _u_num_displ[0] = wp.sin(4.*wp.pi*x) * wp.sin(2.*wp.pi*y) * wp.sin(4.*wp.pi*(t-0.1))
            # _u_num_displ[1] = wp.sin(4.*wp.pi*x) * wp.sin(2.*wp.pi*y) * wp.sin(4.*wp.pi*(t+0.3))
            # _u_num_displ[0] = wp.sin(4.*wp.pi*x) * wp.sin(2.*wp.pi*y) * wp.sin(4.*wp.pi*(t-0.1))
            # _u_num_displ[1] = wp.sin(4.*wp.pi*x) * wp.sin(2.*wp.pi*y) * wp.sin(4.*wp.pi*(t+0.3))
            return _u_num_displ

        # This is the U_num_tilde
        @wp.func
        def U_func(x: wp.float32, y: wp.float32, t: wp.float32):
            _U = _U_vector_vec(0.)
            ux_x = -2.*(x-1./2.) / wp.S_pulse * wp.exp(-((x-1./2.)**2. + (y-1./2.)**2.) / wp.S_pulse)
            uy_y = -2.*(y-1./2.) / wp.S_pulse * wp.exp(-((x-1./2.)**2. + (y-1./2.)**2.) / wp.S_pulse)
            _U[0] = 0. #4.*wp.pi*wp.sin(4.*wp.pi*x)*wp.sin(2.*wp.pi*y)*wp.cos(4.*wp.pi*(t - 1./10.))
            _U[1] = 0. #4.*wp.pi*wp.sin(4.*wp.pi*x)*wp.sin(2.*wp.pi*y)*wp.cos(4.*wp.pi*(t + 3./10.))
            _U[2] = -wp.c_k_led**0.5 * (ux_x + uy_y) #-wp.c_k_led**(1./2.)*(4.*wp.pi*wp.cos(4.*wp.pi*x)*wp.sin(2.*wp.pi*y)*wp.sin(4.*wp.pi*(t - 1./10.)) + 2.*wp.pi*wp.cos(2.*wp.pi*y)*wp.sin(4.*wp.pi*x)*wp.sin(4.*wp.pi*(t + 3./10.)))
            _U[3] = -wp.c_mu_led**0.5 * (ux_x - uy_y) #-wp.c_mu_led**(1./2.)*(4.*wp.pi*wp.cos(4.*wp.pi*x)*wp.sin(2.*wp.pi*y)*wp.sin(4.*wp.pi*(t - 1./10.)) - 2.*wp.pi*wp.cos(2.*wp.pi*y)*wp.sin(4.*wp.pi*x)*wp.sin(4.*wp.pi*(t + 3./10.)))
            _U[4] = -wp.c_mu_led**0.5 * (ux_x + uy_y) #-wp.c_mu_led**(1./2.)*(2.*wp.pi*wp.cos(2.*wp.pi*y)*wp.sin(4.*wp.pi*x)*wp.sin(4.*wp.pi*(t - 1./10.)) + 4.*wp.pi*wp.cos(4.*wp.pi*x)*wp.sin(2.*wp.pi*y)*wp.sin(4.*wp.pi*(t + 3./10.)))
            return _U
        
        @wp.func
        def dUdx_func(x: wp.float32, y: wp.float32, t: wp.float32):
            _dUdx = _U_vector_vec(0.)
            exp_term = wp.exp(-((x-1./2.)**2. + (y-1./2.)**2.)/wp.S_pulse)
            # given the symmetry of the pulse: ux = uy.
            ux_xx = (-2./wp.S_pulse + 4. * (x-1./2.)**2. / wp.S_pulse**2.) * exp_term
            ux_xy = (4. * (x-1./2.) * (y-1./2.) / wp.S_pulse**2.) * exp_term
            _dUdx[0] = 0. 
            _dUdx[1] = 0. 
            _dUdx[2] = -wp.c_k_led**0.5 * (ux_xx + ux_xy)  
            _dUdx[3] = -wp.c_mu_led**0.5 * (ux_xx - ux_xy) 
            _dUdx[4] = -wp.c_mu_led**0.5 * (ux_xx + ux_xy) 
            return _dUdx
        
        @wp.func
        def dUdy_func(x: wp.float32, y: wp.float32, t: wp.float32):
            _dUdy = _U_vector_vec(0.)
            exp_term = wp.exp(-((x-1./2.)**2. + (y-1./2.)**2.)/wp.S_pulse)
            # given the symmetry of the pulse: ux = uy.
            ux_xy = (4. * (x-1./2.) * (y-1./2.) / wp.S_pulse**2.) * exp_term
            ux_yy = (-2./wp.S_pulse + 4. * (y-1./2.)**2. / wp.S_pulse**2.) * exp_term
            _dUdy[0] = 0.
            _dUdy[1] = 0.
            _dUdy[2] = -wp.c_k_led**0.5 * (ux_xy + ux_yy)
            _dUdy[3] = -wp.c_mu_led**0.5 * (ux_xy - ux_yy)
            _dUdy[4] = -wp.c_mu_led**0.5 * (ux_xy + ux_yy)
            return _dUdy

        @wp.func
        def B_func(x: wp.float32, y: wp.float32, t: wp.float32):
            _B = _U_vector_vec(0.)
            _B[0] = 0. 
            _B[1] = 0. 
            _B[2] = 0.0
            _B[3] = 0.0
            _B[4] = 0.0
            return _B
        
        @wp.func
        def phi_x_lb_func(_U: Any):
            _phi_x = _U_vector_vec(0.)
            _phi_x[0] = (wp.c_k_led**0.5*_U[2]+wp.c_mu_led**0.5*_U[3])/wp.c_led
            _phi_x[1] = wp.c_mu_led**0.5*_U[4]/wp.c_led
            _phi_x[2] = wp.c_k_led**0.5*_U[0]/wp.c_led
            _phi_x[3] = wp.c_mu_led**0.5*_U[0]/wp.c_led
            _phi_x[4] = wp.c_mu_led**0.5*_U[1]/wp.c_led
            return _phi_x

        @wp.func
        def phi_y_lb_func(_U: Any):
            _phi_y = _U_vector_vec(0.)
            _phi_y[0] = wp.c_mu_led**0.5*_U[4]/wp.c_led
            _phi_y[1] = (wp.c_k_led**0.5*_U[2]-wp.c_mu_led**0.5*_U[3])/wp.c_led
            _phi_y[2] = wp.c_k_led**0.5*_U[1]/wp.c_led
            _phi_y[3] = -wp.c_mu_led**0.5*_U[1]/wp.c_led
            _phi_y[4] = wp.c_mu_led**0.5*_U[0]/wp.c_led
            return _phi_y
        
            
        #2nd order IC with relatively high error due to precision issues
        @wp.kernel
        def initial_conditions_v2_kernel(f: wp.array4d(dtype=Any), U_num_tilde: wp.array4d(dtype=Any), u_num_displ_out: wp.array4d(dtype=Any)):
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)
            x = (self.compute_dtype(index[0]) + self.compute_dtype(0.5)) * wp.delta_x_led
            y = (self.compute_dtype(index[1]) + self.compute_dtype(0.5)) * wp.delta_x_led
            t = self.compute_dtype(0.0)
            #evaluate relevant properties from analytical solutions
            _u_num_displ = u_num_displ_func(x, y, t)
            _U = U_func(x, y, t)
            _B = B_func(x, y, t)*wp.delta_t_led

            # for i in range(2):
            #     _U[i] += _B[i] * wp.float32(0.5)

            _dUdx = dUdx_func(x, y, t)*wp.delta_x_led
            _dUdy = dUdy_func(x, y, t)*wp.delta_x_led
            _f = _f_vector_vec()
            _f0 = wp.float32(0.25)*((_U+wp.float32(wp.float32(2.0))*phi_x_lb_func(_U)) + wp.float32(0.5)*(-(_B+wp.float32(wp.float32(2.0))*phi_x_lb_func(_B)) - _dUdx - phi_x_lb_func(_dUdx) + phi_y_lb_func(_dUdy) + wp.float32(wp.float32(2.0))*(phi_x_lb_func(phi_x_lb_func(_dUdx)+phi_y_lb_func(_dUdy)))))
            _f1 = wp.float32(0.25)*((_U+wp.float32(wp.float32(2.0))*phi_y_lb_func(_U)) + wp.float32(0.5)*(-(_B+wp.float32(wp.float32(2.0))*phi_y_lb_func(_B)) - _dUdy + phi_x_lb_func(_dUdx) - phi_y_lb_func(_dUdy) + wp.float32(wp.float32(2.0))*(phi_y_lb_func(phi_x_lb_func(_dUdx)+phi_y_lb_func(_dUdy)))))
            _f2 = wp.float32(0.25)*((_U-wp.float32(wp.float32(2.0))*phi_x_lb_func(_U)) + wp.float32(0.5)*(-(_B-wp.float32(wp.float32(2.0))*phi_x_lb_func(_B)) + _dUdx - phi_x_lb_func(_dUdx) + phi_y_lb_func(_dUdy) - wp.float32(wp.float32(2.0))*(phi_x_lb_func(phi_x_lb_func(_dUdx)+phi_y_lb_func(_dUdy)))))
            _f3 = wp.float32(0.25)*((_U-wp.float32(wp.float32(2.0))*phi_y_lb_func(_U)) + wp.float32(0.5)*(-(_B-wp.float32(wp.float32(2.0))*phi_y_lb_func(_B)) + _dUdy + phi_x_lb_func(_dUdx) - phi_y_lb_func(_dUdy) - wp.float32(wp.float32(2.0))*(phi_y_lb_func(phi_x_lb_func(_dUdx)+phi_y_lb_func(_dUdy)))))
            
            # # dir x
            # _f0 = 0.25*((_U+2.0*phi_x_lb(_U)) + 0.5*(-(_B+2.0*phi_x_lb(_B)) - _dUdx - phi_x_lb(_dUdx) + phi_y_lb(_dUdy) + 2.0*(phi_x_lb(phi_x_lb(_dUdx)+phi_y_lb(_dUdy)))))
            # #  dir y
            # _f1 = 0.25*((_U+2.0*phi_y_lb(_U)) + 0.5*(-(_B+2.0*phi_y_lb(_B)) - _dUdy + phi_x_lb(_dUdx) - phi_y_lb(_dUdy) + 2.0*(phi_y_lb(phi_x_lb(_dUdx)+phi_y_lb(_dUdy)))))
            # # dir -x
            # _f2 = 0.25*((_U-2.0*phi_x_lb(_U)) + 0.5*(-(_B-2.0*phi_x_lb(_B)) + _dUdx - phi_x_lb(_dUdx) + phi_y_lb(_dUdy) - 2.0*(phi_x_lb(phi_x_lb(_dUdx)+phi_y_lb(_dUdy)))))
            # # dir -y
            # _f3 = 0.25*((_U-2.0*phi_y_lb(_U)) + 0.5*(-(_B-2.0*phi_y_lb(_B)) + _dUdy + phi_x_lb(_dUdx) - phi_y_lb(_dUdy) - 2.0*(phi_y_lb(phi_x_lb(_dUdx)+phi_y_lb(_dUdy)))))
            for s in range(5):
                _f[s] = _f0[s]
                _f[s + 5] = _f1[s]
                _f[s + 10] = _f2[s]
                _f[s + 15] = _f3[s]
            
            for l in range(20):
                f[l, index[0], index[1], index[2]] = self.store_dtype(_f[l])
            for l in range(5):
                U_num_tilde[l, index[0], index[1], index[2]] = self.store_dtype(_U[l])
            for l in range(2):
                u_num_displ_out[l, index[0], index[1], index[2]] = self.store_dtype(_u_num_displ[l])
        
        return None, initial_conditions_v2_kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, U_num_tilde, u_num_displ_out):
        wp.launch(
            self.warp_kernel,
            inputs=[f, U_num_tilde, u_num_displ_out],
            dim=U_num_tilde.shape[1:],
        )
        return f, U_num_tilde, u_num_displ_out
   
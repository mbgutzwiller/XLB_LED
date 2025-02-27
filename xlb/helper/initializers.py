from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.equilibrium import Equilibrium_LED
from xlb.operator.operator import Operator
import warp as wp
from typing import Any



def initialize_eq(f, grid, velocity_set, precision_policy, compute_backend, rho=None, u=None):
    if rho is None:
        rho = grid.create_field(cardinality=1, fill_value=1.0, dtype=precision_policy.compute_precision)
    if u is None:
        u = grid.create_field(cardinality=velocity_set.d, fill_value=0.0, dtype=precision_policy.compute_precision)
    equilibrium = QuadraticEquilibrium()

    if compute_backend == ComputeBackend.JAX:
        f = equilibrium(rho, u)

    elif compute_backend == ComputeBackend.WARP:
        f = equilibrium(rho, u, f)

    del rho, u

    return f

def initialize_f_U_num_LED(f, grid, precision_policy, compute_backend):
    if f is None:
        U_0 = grid.create_field(cardinality=20, fill_value=0.0, dtype=precision_policy.compute_precision)
    
    equilibrium = Equilibrium_LED()

    if compute_backend == ComputeBackend.JAX:
        f = equilibrium(U_0)  # TODO

    elif compute_backend == ComputeBackend.WARP:
        U_0 = grid.create_field(cardinality=5, fill_value=0.0, dtype=precision_policy.compute_precision)
        u_num_displ = grid.create_field(cardinality=5, fill_value=0.0, dtype=precision_policy.compute_precision)
        f = equilibrium(U_0, f)
    return f, U_0, u_num_displ

class Initializer_LED(Operator):
    def __init__(self, velocity_set=None, precision_policy=None, compute_backend=None):
        super().__init__(velocity_set, precision_policy, compute_backend)
    
    # def initialize_eq_LED(self, f, grid, precision_policy, compute_backend):
    #     if f is None:
    #         U_0 = grid.create_field(cardinality=20, fill_value=0.0, dtype=precision_policy.compute_precision)
        
    #     equilibrium = Equilibrium_LED()

    #     if compute_backend == ComputeBackend.JAX:
    #         f = equilibrium(U_0)  # TODO

    #     elif compute_backend == ComputeBackend.WARP:
    #         U_0 = grid.create_field(cardinality=5, fill_value=0.0, dtype=precision_policy.compute_precision)
    #         f = equilibrium(U_0, f)
    #     return f, U_0
    
    def _construct_warp(self):
        _u_num_displ_vector_vec = wp.vec(2, dtype=self.compute_dtype)
        _U_vector_vec = wp.vec(5, dtype=self.compute_dtype)
        _f_vector_vec = wp.vec(20, dtype=self.compute_dtype)

        # Analytical functions to set initial f correctly (non trivial in contrast to fluid LBM)
        @wp.func
        def u_num_displ(x: wp.float32, y: wp.float32, t: wp.float32):
            _u_num_displ = _u_num_displ_vector_vec()
            _u_num_displ[0] = wp.sin(4.*wp.pi*(x-0.3*t)) * wp.cos(2.*wp.pi*(y-0.8*t)) * wp.sin(4.*wp.pi*(t-0.1))
            _u_num_displ[1] = wp.cos(4.*wp.pi*(x-0.7*t)) * wp.sin(2.*wp.pi*(y-0.1*t)) * wp.cos(4.*wp.pi*(t+0.4))
            return _u_num_displ

        # This is the U_num_tilde, not the displacement.
        @wp.func
        def U(x: wp.float32, y: wp.float32, t: wp.float32):
            U = _U_vector_vec()
            U[0] = 1.6*wp.pi*wp.sin(wp.pi*(-1.6*t + 2.0*y))*wp.sin(wp.pi*(-1.2*t + 4.0*x))*wp.sin(wp.pi*(4.0*t - 0.4)) + 4.0*wp.pi*wp.sin(wp.pi*(-1.2*t + 4.0*x))*wp.cos(wp.pi*(-1.6*t + 2.0*y))*wp.cos(wp.pi*(4.0*t - 0.4)) - 1.2*wp.pi*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.6*t + 2.0*y))*wp.cos(wp.pi*(-1.2*t + 4.0*x))
            U[1] = 2.8*wp.pi*wp.sin(wp.pi*(-2.8*t + 4.0*x))*wp.sin(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(4.0*t + 1.6)) - 4.0*wp.pi*wp.sin(wp.pi*(-0.2*t + 2.0*y))*wp.sin(wp.pi*(4.0*t + 1.6))*wp.cos(wp.pi*(-2.8*t + 4.0*x)) - 0.2*wp.pi*wp.cos(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(4.0*t + 1.6))
            U[2] = -wp.c_k_led*(4.0*wp.pi*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.6*t + 2.0*y))*wp.cos(wp.pi*(-1.2*t + 4.0*x))+2.0*wp.pi*wp.cos(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(4.0*t + 1.6)))
            U[3] = -wp.c_mu_led*(4.0*wp.pi*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.6*t + 2.0*y))*wp.cos(wp.pi*(-1.2*t + 4.0*x))-2.0*wp.pi*wp.cos(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(4.0*t + 1.6)))
            U[4] = -wp.c_mu_led*(-2.0*wp.pi*wp.sin(wp.pi*(-1.6*t + 2.0*y))*wp.sin(wp.pi*(-1.2*t + 4.0*x))*wp.sin(wp.pi*(4.0*t - 0.4))-4.0*wp.pi*wp.sin(wp.pi*(-2.8*t + 4.0*x))*wp.sin(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(4.0*t + 1.6)))
            return U
        
        @wp.func
        def dUdx(x: wp.float32, y: wp.float32, t: wp.float32):
            dUdx = _U_vector_vec()
            dUdx[0] = 6.4*wp.pi**2.0*wp.sin(wp.pi*(-1.6*t + 2.0*y))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.2*t + 4.0*x)) + 4.8*wp.pi**2.0*wp.sin(wp.pi*(-1.2*t + 4.0*x))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.6*t + 2.0*y)) + 16.0*wp.pi**2.0*wp.cos(wp.pi*(-1.6*t + 2.0*y))*wp.cos(wp.pi*(-1.2*t + 4.0*x))*wp.cos(wp.pi*(4.0*t - 0.4))
            dUdx[1] = 16.0*wp.pi**2.0*wp.sin(wp.pi*(-2.8*t + 4.0*x))*wp.sin(wp.pi*(-0.2*t + 2.0*y))*wp.sin(wp.pi*(4.0*t + 1.6)) + 0.8*wp.pi**2.0*wp.sin(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(4.0*t + 1.6)) + 11.2*wp.pi**2.0*wp.sin(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(4.0*t + 1.6))
            dUdx[2] = -wp.c_k_led*(-8.0*wp.pi**2.0*wp.sin(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(4.0*t + 1.6)) - 16.0*wp.pi**2.0*wp.sin(wp.pi*(-1.2*t + 4.0*x))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.6*t + 2.0*y)))
            dUdx[3] = -wp.c_mu_led*(8.0*wp.pi**2.0*wp.sin(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(4.0*t + 1.6)) - 16.0*wp.pi**2.0*wp.sin(wp.pi*(-1.2*t + 4.0*x))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.6*t + 2.0*y))) 
            dUdx[4] = -wp.c_mu_led*(-8.0*wp.pi**2.0*wp.sin(wp.pi*(-1.6*t + 2.0*y))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.2*t + 4.0*x)) - 16.0*wp.pi**2.0*wp.sin(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(4.0*t + 1.6)))
            return dUdx
        
        @wp.func
        def dUdy(x: wp.float32, y: wp.float32, t: wp.float32):
            dUdy = _U_vector_vec()
            dUdy[0] = -8.0*wp.pi**2.0*wp.sin(wp.pi*(-1.6*t + 2.0*y))*wp.sin(wp.pi*(-1.2*t + 4.0*x))*wp.cos(wp.pi*(4.0*t - 0.4)) + 2.4*wp.pi**2.0*wp.sin(wp.pi*(-1.6*t + 2.0*y))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.2*t + 4.0*x)) + 3.2*wp.pi**2.0*wp.sin(wp.pi*(-1.2*t + 4.0*x))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.6*t + 2.0*y))
            dUdy[1] = 5.6*wp.pi**2.0*wp.sin(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(4.0*t + 1.6)) + 0.4*wp.pi**2.0*wp.sin(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(4.0*t + 1.6)) - 8.0*wp.pi**2.0*wp.sin(wp.pi*(4.0*t + 1.6))*wp.cos(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(-0.2*t + 2.0*y))
            dUdy[2] = -wp.c_k_led*(-8.0*wp.pi**2.0*wp.sin(wp.pi*(-1.6*t + 2.0*y))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.2*t + 4.0*x)) - 4.0*wp.pi**2.0*wp.sin(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(4.0*t + 1.6)))
            dUdy[3] = -wp.c_mu_led*(-8.0*wp.pi**2.0*wp.sin(wp.pi*(-1.6*t + 2.0*y))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.2*t + 4.0*x)) + 4.0*wp.pi**2.0*wp.sin(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(4.0*t + 1.6)))
            dUdy[4] = -wp.c_mu_led*(-8.0*wp.pi**2.0*wp.sin(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(4.0*t + 1.6)) - 4.0*wp.pi**2.0*wp.sin(wp.pi*(-1.2*t + 4.0*x))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.6*t + 2.0*y)))
            return dUdy

        @wp.func
        def B(x: wp.float32, y: wp.float32, t: wp.float32):
            B = _U_vector_vec()
            B[0] = -wp.c_k_led**2.0*(-8.0*wp.pi**2.0*wp.sin(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(4.0*t + 1.6)) - 16.0*wp.pi**2.0*wp.sin(wp.pi*(-1.2*t + 4.0*x))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.6*t + 2.0*y))) - wp.c_mu_led**2.0*(-8.0*wp.pi**2.0*wp.sin(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(4.0*t + 1.6)) - 4.0*wp.pi**2.0*wp.sin(wp.pi*(-1.2*t + 4.0*x))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.6*t + 2.0*y))) - wp.c_mu_led**2.0*(8.0*wp.pi**2.0*wp.sin(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(4.0*t + 1.6)) - 16.0*wp.pi**2.0*wp.sin(wp.pi*(-1.2*t + 4.0*x))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.6*t + 2.0*y))) + 12.8*wp.pi**2.0*wp.sin(wp.pi*(-1.6*t + 2.0*y))*wp.sin(wp.pi*(-1.2*t + 4.0*x))*wp.cos(wp.pi*(4.0*t - 0.4)) - 3.84*wp.pi**2.0*wp.sin(wp.pi*(-1.6*t + 2.0*y))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.2*t + 4.0*x)) - 20.0*wp.pi**2.0*wp.sin(wp.pi*(-1.2*t + 4.0*x))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.6*t + 2.0*y)) - 9.6*wp.pi**2.0*wp.cos(wp.pi*(-1.6*t + 2.0*y))*wp.cos(wp.pi*(-1.2*t + 4.0*x))*wp.cos(wp.pi*(4.0*t - 0.4))
            B[1] = -wp.c_k_led**2.0*(-8.0*wp.pi**2.0*wp.sin(wp.pi*(-1.6*t + 2.0*y))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.2*t + 4.0*x)) - 4.0*wp.pi**2.0*wp.sin(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(4.0*t + 1.6))) - wp.c_mu_led**2.0*(-8.0*wp.pi**2.0*wp.sin(wp.pi*(-1.6*t + 2.0*y))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.2*t + 4.0*x)) - 16.0*wp.pi**2.0*wp.sin(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(4.0*t + 1.6))) + wp.c_mu_led**2.0*(-8.0*wp.pi**2.0*wp.sin(wp.pi*(-1.6*t + 2.0*y))*wp.sin(wp.pi*(4.0*t - 0.4))*wp.cos(wp.pi*(-1.2*t + 4.0*x)) + 4.0*wp.pi**2.0*wp.sin(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(4.0*t + 1.6))) - 22.4*wp.pi**2.0*wp.sin(wp.pi*(-2.8*t + 4.0*x))*wp.sin(wp.pi*(-0.2*t + 2.0*y))*wp.sin(wp.pi*(4.0*t + 1.6)) - 1.12*wp.pi**2.0*wp.sin(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(4.0*t + 1.6)) - 23.88*wp.pi**2.0*wp.sin(wp.pi*(-0.2*t + 2.0*y))*wp.cos(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(4.0*t + 1.6)) + 1.6*wp.pi**2.0*wp.sin(wp.pi*(4.0*t + 1.6))*wp.cos(wp.pi*(-2.8*t + 4.0*x))*wp.cos(wp.pi*(-0.2*t + 2.0*y))
            B[2] = 0.0
            B[3] = 0.0
            B[4] = 0.0
            return B
        
        @wp.func
        def phi_x_lb(_U: Any):
            phi_x = _U_vector_vec()
            phi_x[0] = (wp.c_k_led*_U[2]+wp.c_mu_led*_U[3])/wp.c_led
            phi_x[1] = wp.c_mu_led*_U[4]/wp.c_led
            phi_x[2] = wp.c_k_led*_U[0]/wp.c_led
            phi_x[3] = wp.c_mu_led*_U[0]/wp.c_led
            phi_x[4] = wp.c_mu_led*_U[1]/wp.c_led
            return phi_x

        @wp.func
        def phi_y_lb(_U: Any):
            phi_y = _U_vector_vec()
            phi_y[0] = wp.c_mu_led*_U[4]/wp.c_led
            phi_y[1] = (wp.c_k_led*_U[2]-wp.c_mu_led*_U[3])/wp.c_led
            phi_y[2] = wp.c_k_led*_U[1]/wp.c_led
            phi_y[3] = -wp.c_mu_led*_U[1]/wp.c_led
            phi_y[4] = wp.c_mu_led*_U[0]/wp.c_led
            return phi_y
        
            
        #2nd order IC with relatively high error due to precision issues
        @wp.kernel
        def initial_conditions_v2_kernel(U_num_tilde: wp.array4d(dtype=Any), f: wp.array4d(dtype=Any), u_num_displ_out: wp.array4d(dtype=Any)):
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)
            x = (self.compute_dtype(index[0]) + self.compute_dtype(0.5)) * wp.delta_x_led
            y = (self.compute_dtype(index[1]) + self.compute_dtype(0.5)) * wp.delta_x_led
            t = self.compute_dtype(0.0)
            #evaluate relevant properties from analytical solutions
            _u_num_displ = u_num_displ(x, y, t)
            _U = U(x, y, t)
            _B = B(x, y, t)*wp.delta_t_led
            _dUdx = dUdx(x, y, t)*wp.delta_x_led
            _dUdy = dUdy(x, y, t)*wp.delta_x_led
            _f = _f_vector_vec()
            _f0 = 0.25*((_U+2.0*phi_x_lb(_U)) + 0.5*(-(_B+2.0*phi_x_lb(_B)) - _dUdx - phi_x_lb(_dUdx) + phi_y_lb(_dUdy) + 2.0*(phi_x_lb(phi_x_lb(_dUdx)+phi_y_lb(_dUdy)))))
            _f1 = 0.25*((_U+2.0*phi_y_lb(_U)) + 0.5*(-(_B+2.0*phi_y_lb(_B)) - _dUdy + phi_x_lb(_dUdx) - phi_y_lb(_dUdy) + 2.0*(phi_y_lb(phi_x_lb(_dUdx)+phi_y_lb(_dUdy)))))
            _f2 = 0.25*((_U-2.0*phi_x_lb(_U)) + 0.5*(-(_B-2.0*phi_x_lb(_B)) + _dUdx - phi_x_lb(_dUdx) + phi_y_lb(_dUdy) - 2.0*(phi_x_lb(phi_x_lb(_dUdx)+phi_y_lb(_dUdy)))))
            _f3 = 0.25*((_U-2.0*phi_y_lb(_U)) + 0.5*(-(_B-2.0*phi_y_lb(_B)) + _dUdy + phi_x_lb(_dUdx) - phi_y_lb(_dUdy) - 2.0*(phi_y_lb(phi_x_lb(_dUdx)+phi_y_lb(_dUdy)))))
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
   
# import jax.numpy as jnp
# from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator import Operator
# from functools import partial


class Error_LED_u(Operator):
    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _u_num_vec = wp.vec(2, dtype=self.compute_dtype)
        _x_y_vec = wp.vec(2, dtype=self.compute_dtype)
        
        # Construct the functional
        @wp.func
        def functional(uxy_num: _u_num_vec, index: wp.vec3i, t: wp.float32):
            # Target manufactured solution
            loc = _x_y_vec()
            _uxy_ex = _u_num_vec()
            loc[0] = (self.compute_dtype(index[0]) + self.compute_dtype(0.5)) * wp.delta_x_led
            loc[1] = (self.compute_dtype(index[1]) + self.compute_dtype(0.5)) * wp.delta_x_led
            _uxy_ex[0] = wp.sin(4.*wp.pi*(loc[0]-0.3*t)) * wp.cos(2.*wp.pi*(loc[1]-0.8*t)) * wp.sin(4.*wp.pi*(t-0.1))
            _uxy_ex[1] = wp.cos(4.*wp.pi*(loc[0]-0.7*t)) * wp.sin(2.*wp.pi*(loc[1]-0.1*t)) * wp.cos(4.*wp.pi*(t+0.4))
            
            err_x = _uxy_ex[0] - uxy_num[0]
            err_y = _uxy_ex[1] - uxy_num[1]
            
            err_sq = err_x * err_x + err_y * err_y
            # this is for linf norm
            # err_vec = _u_num_vec()
            # for i in range(2):
                # err_vec[i] = wp.abs(uxy_num[i] - _u_ex[i])
            return err_sq
        
        
        # Construct the warp kernel
        @wp.kernel
        def kernel(
            u_num_displ: wp.array4d(dtype=wp.float32),
            error_out: wp.array4d(dtype=wp.float32),
            timestep: wp.float32
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)  # TODO: Warp needs to fix this

            _uxy_num = _u_num_vec()
            for l in range(2):
                _uxy_num[l] = u_num_displ[l, index[0], index[1], index[2]]
            
            t = self.compute_dtype(timestep) * wp.delta_t_led

            err_sq = functional(_uxy_num, index, t)

            error_out[0, index[0], index[1], index[2]] = self.store_dtype(err_sq)
        
        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, u_num_displ, error_out, timestep):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                u_num_displ,
                error_out,
                timestep
            ],
            dim=u_num_displ.shape[1:],
        )
        return error_out

# import jax.numpy as jnp
# from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator import Operator
# from functools import partial


class Sum_Error_LED_u(Operator):
    def _construct_warp(self):
        
        # Construct the warp kernel
        @wp.kernel
        def kernel(
            error: wp.array4d(dtype=Any),
            result: wp.array(dtype=Any)
        ):
            i, j, k = wp.tid()
            val = error[0, i, j, k]
            wp.atomic_add(result, 0, val)

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, error, result):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                error,
                result,
            ],
            dim=error.shape,
        )

        return result

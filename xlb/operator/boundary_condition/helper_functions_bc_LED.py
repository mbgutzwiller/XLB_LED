from xlb import DefaultConfig, ComputeBackend
import warp as wp
from typing import Any


class HelperFunctionsBC_LED(object):
    def __init__(self, velocity_set=None, precision_policy=None, compute_backend=None):
        if compute_backend == ComputeBackend.JAX:
            raise ValueError("This helper class contains helper functions only for the WARP implementation of some BCs not JAX!")

        # Set the default values from the global config
        self.velocity_set = velocity_set or DefaultConfig.velocity_set
        self.precision_policy = precision_policy or DefaultConfig.default_precision_policy
        self.compute_backend = compute_backend or DefaultConfig.default_backend

        # Set the compute and Store dtypes
        compute_dtype = self.precision_policy.compute_precision.wp_dtype
        store_dtype = self.precision_policy.store_precision.wp_dtype

        # Set local constants
        _d = self.velocity_set.d
        _q = self.velocity_set.q
        _opp_indices = self.velocity_set.opp_indices
        _w = self.velocity_set.w
        _c = self.velocity_set.c
        _c_float = self.velocity_set.c_float
        _qi = self.velocity_set.qi
        _u_vec = wp.vec(5, dtype=compute_dtype)
        _f_vec = wp.vec(20, dtype=compute_dtype)
        _missing_mask_vec = wp.vec(4, dtype=wp.uint8)  # TODO fix vec bool

        # Define the operator needed for computing the momentum flux
        # momentum_flux = MomentumFlux(velocity_set, precision_policy, compute_backend)

        @wp.func
        def get_thread_data(
            f_pre: wp.array4d(dtype=Any),
            f_post: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
            index: wp.vec3i,
        ):
            # Get the boundary id and missing mask
            _f_pre = _f_vec()
            _f_post = _f_vec()
            _boundary_id = bc_mask[0, index[0], index[1], index[2]]
            _missing_mask = _missing_mask_vec()
            for l in range(4):
                # q-sized vector of populations
                for m in range(5):
                    _f_pre[l * 5 + m] = compute_dtype(f_pre[l * 5 + m, index[0], index[1], index[2]])
                    _f_post[l * 5 + m] = compute_dtype(f_post[l * 5 + m, index[0], index[1], index[2]])

                # TODO fix vec bool
                if missing_mask[l, index[0], index[1], index[2]]:
                    _missing_mask[l] = wp.uint8(1)
                else:
                    _missing_mask[l] = wp.uint8(0)

            return _f_pre, _f_post, _boundary_id, _missing_mask

        self.get_thread_data = get_thread_data

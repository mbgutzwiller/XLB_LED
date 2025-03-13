from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium.equilibrium import Equilibrium
from xlb.operator import Operator


class Equilibrium_LED(Equilibrium):
    # def __init__(self, velocity_set, precision_policy, compute_backend):
    #     super().__init__(velocity_set, precision_policy, compute_backend)

    #     self.K = 137.8e9
    #     self.rho = 8.96
    #     self.nu = 0.343
    #     self.mu = (3 * self.K * (1 - 2 * self.nu)) / (2 * (1 + self.nu))

    #     self.c_mu = self.compute_dtype((self.mu / self.rho) ** 0.5)
    #     self.c_K = self.compute_dtype((self.K / self.rho) ** 0.5)
    #     self.c = self.compute_dtype(1)

    """
    Equilibrium function as defined in equation (31) in paper by Oliver et al.
    """

    # @Operator.register_backend(ComputeBackend.JAX)
    # @partial(jit, static_argnums=(0))
    # def jax_implementation(self, U_num_tilde):
    #     # Matching velocity set (f_10, f_01, f_(-10), f_(0-1))
    #     # U_num_tilde is the U_num with forcing.
    #     c_K_j_s = U_num_tilde[2] * self.c_K
    #     c_mu_j_d = U_num_tilde[3] * self.c_mu
    #     c_mu_j_xy = U_num_tilde[4] * self.c_mu
    #     c_mu_v_x = U_num_tilde[0] * self.c_mu
    #     c_mu_v_y = U_num_tilde[1] * self.c_mu

    #     phi_x_tilde_2c = jnp.zeros_like(U_num_tilde)
    #     phi_y_tilde_2c = jnp.zeros_like(U_num_tilde)

    #     phi_x_tilde_2c = phi_x_tilde_2c.at[0].set(c_K_j_s + c_mu_j_d)
    #     phi_x_tilde_2c = phi_x_tilde_2c.at[1].set(c_mu_j_xy)
    #     phi_x_tilde_2c = phi_x_tilde_2c.at[2].set(self.c_K * U_num_tilde[0])
    #     phi_x_tilde_2c = phi_x_tilde_2c.at[3].set(c_mu_v_x)
    #     phi_x_tilde_2c = phi_x_tilde_2c.at[4].set(c_mu_v_y)

    #     phi_y_tilde_2c = phi_y_tilde_2c.at[0].set(c_mu_j_xy)
    #     phi_y_tilde_2c = phi_y_tilde_2c.at[1].set(c_K_j_s - c_mu_j_d)
    #     phi_y_tilde_2c = phi_y_tilde_2c.at[2].set(self.c_K * U_num_tilde[1])
    #     phi_y_tilde_2c = phi_y_tilde_2c.at[3].set(-c_mu_v_y)
    #     phi_y_tilde_2c = phi_y_tilde_2c.at[4].set(c_mu_v_x)

    #     phi_x_tilde_2c *= 2/self.c
    #     phi_y_tilde_2c *= 2/self.c

    #     f_eq = jnp.zeros(20)

    #     f_eq = f_eq.at[0:5].set(U_num_tilde + phi_x_tilde_2c)
    #     f_eq = f_eq.at[5:10].set(U_num_tilde + phi_y_tilde_2c)
    #     f_eq = f_eq.at[10:15].set(U_num_tilde - phi_x_tilde_2c)
    #     f_eq = f_eq.at[15:20].set(U_num_tilde - phi_y_tilde_2c)

    #     f_eq *= 0.25

    #     # cu = 3.0 * jnp.tensordot(self.velocity_set.c, U_num_tilde, axes=(0, 0))
    #     # usqr = 1.5 * jnp.sum(jnp.square(U_num_tilde), axis=0, keepdims=True)
    #     # w = self.velocity_set.w.reshape((-1,) + (1,) * (len(rho.shape) - 1))
    #     # feq = rho * w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)
    #     return f_eq

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update

        scale_2 = wp.constant(self.compute_dtype(2.0))
        scale_025 = self.compute_dtype(0.25)

        _f_vec = wp.vec(20, dtype=self.compute_dtype)
        _U_num_tilde_vec = wp.vec(5, dtype=self.compute_dtype)
        _phi_x_tilde_2c = wp.vec(5, dtype=self.compute_dtype)
        _phi_y_tilde_2c = wp.vec(5, dtype=self.compute_dtype)

        # Construct the equilibrium functional
        @wp.func
        def functional(
            U_num_tilde: Any
        ):
            c_K_j_s = U_num_tilde[2] * wp.c_k_led ** 0.5
            c_mu_j_d = U_num_tilde[3] * wp.c_mu_led ** 0.5
            c_mu_j_xy = U_num_tilde[4] * wp.c_mu_led ** 0.5
            c_mu_v_x = U_num_tilde[0] * wp.c_mu_led ** 0.5
            c_mu_v_y = U_num_tilde[1] * wp.c_mu_led ** 0.5
            phi_x_tilde = _phi_x_tilde_2c()
            phi_y_tilde = _phi_y_tilde_2c()

            phi_x_tilde[0] = c_K_j_s + c_mu_j_d
            phi_x_tilde[1] = c_mu_j_xy
            phi_x_tilde[2] = wp.c_k_led ** 0.5 * U_num_tilde[0]
            phi_x_tilde[3] = c_mu_v_x
            phi_x_tilde[4] = c_mu_v_y

            phi_y_tilde[0] = c_mu_j_xy
            phi_y_tilde[1] = c_K_j_s - c_mu_j_d
            phi_y_tilde[2] = wp.c_k_led ** 0.5 * U_num_tilde[1]
            phi_y_tilde[3] = -c_mu_v_y
            phi_y_tilde[4] = c_mu_v_x

            phi_x_tilde_c = phi_x_tilde / wp.c_led
            phi_y_tilde_c = phi_y_tilde / wp.c_led

            f_eq = _f_vec()
            for i in range(5):
                f_eq[i] = scale_025 * (U_num_tilde[i] + scale_2 * phi_x_tilde_c[i])
                f_eq[i + 5] = scale_025 * (U_num_tilde[i] + scale_2 * phi_y_tilde_c[i])
                f_eq[i + 10] = scale_025* (U_num_tilde[i] - scale_2 * phi_x_tilde_c[i])
                f_eq[i + 15] = scale_025 * (U_num_tilde[i] - scale_2 * phi_y_tilde_c[i])

            return f_eq

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            U_num_tilde: wp.array4d(dtype=Any),
            f: wp.array4d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Get the equilibrium
            _U_num_tilde = _U_num_tilde_vec()
            for d in range(5):
                _U_num_tilde[d] = U_num_tilde[d, index[0], index[1], index[2]]
            f_eq = functional(_U_num_tilde)

            # Set the output
            for l in range(20):
                f[l, index[0], index[1], index[2]] = self.store_dtype(f_eq[l])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, U_num_tilde, f):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                U_num_tilde,
                f,
            ],
            dim=U_num_tilde.shape[1:],
        )
        return f

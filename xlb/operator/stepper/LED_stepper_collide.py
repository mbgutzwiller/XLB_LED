# Base class for all stepper operators

from functools import partial
from jax import jit
import warp as wp
from typing import Any

from xlb import DefaultConfig
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import Precision
from xlb.operator import Operator
from xlb.operator.stream import Stream_LED
from xlb.operator.collision import BGK_LED
from xlb.operator.equilibrium import Equilibrium_LED
from xlb.operator.macroscopic import Macroscopic_LED
from xlb.operator.displacement.displacement_LED import Displacement_LED
from xlb.operator.stepper import Stepper
from xlb.operator.boundary_condition.boundary_condition import ImplementationStep
from xlb.operator.boundary_condition.boundary_condition_registry import boundary_condition_registry
from xlb.operator.collision import ForcedCollision
from xlb.operator.boundary_masker import IndicesBoundaryMasker, MeshBoundaryMasker
from xlb.helper import check_bc_overlaps
from xlb.helper.LED_solver_v2 import create_LED_fields


class LinearElastodynamicsStepperCollide(Stepper):
    def __init__(
        self,
        grid,
        boundary_conditions=[],
        collision_type="BGK_LED",
        forcing_scheme="exact_difference",
        force_vector=None,
    ):
        super().__init__(grid, boundary_conditions)

        # Construct the collision operator, using equation for collision (28) 
        if collision_type == "BGK_LED":
            self.collision_LED = BGK_LED(self.velocity_set, self.precision_policy, self.compute_backend)
        elif collision_type == "KBC":
            raise NotImplementedError
            # self.collision = KBC(self.velocity_set, self.precision_policy, self.compute_backend)

        if force_vector is not None:
            raise NotImplementedError
            # self.collision = ForcedCollision(collision_operator=self.collision, forcing_scheme=forcing_scheme, force_vector=force_vector)

        # Construct the operators
        self.stream_LED = Stream_LED(self.velocity_set, self.precision_policy, self.compute_backend)
        self.equilibrium_LED = Equilibrium_LED(self.velocity_set, self.precision_policy, self.compute_backend)
        self.macroscopic_LED = Macroscopic_LED(self.velocity_set, self.precision_policy, self.compute_backend)
        self.displacement_LED = Displacement_LED((self.velocity_set, self.precision_policy, self.compute_backend))

    def prepare_fields(self, initializer=None):  # TODO: initialize the field, add initialize_eq to helper.initializers
        """Prepare the fields required for the stepper.

        Args:
            initializer: Optional operator to initialize the distribution functions.
                        If provided, it should be a callable that takes (grid, velocity_set,
                        precision_policy, compute_backend) as arguments and returns initialized f_0.
                        If None, default equilibrium initialization is used with rho=1 and u=0.

        Returns:
            Tuple of (f_0, f_1, bc_mask, missing_mask):
                - f_0: Initial distribution functions
                - f_1: Copy of f_0 for double-buffering
                - bc_mask: Boundary condition mask indicating which BC applies to each node
                - missing_mask: Mask indicating which populations are missing at boundary nodes
        """
        # Create fields using the helper function
        _, f_0, f_1, f_temp, missing_mask, bc_mask = create_LED_fields(
            grid=self.grid, compute_backend=self.compute_backend, precision_policy=self.precision_policy
        )

        # Initialize distribution functions if initializer is provided
        # if initializer is not None:
        #     f_0 = initializer(self.grid, self.velocity_set, self.precision_policy, self.compute_backend)
        
        from xlb.helper.initializers_v2 import initialize_f_U_num_LED  # TODO: initialze eq for LED

        f_0 , U_0, u_num_displ_0, u_num_displ_1 = initialize_f_U_num_LED(f_0, self.grid, self.precision_policy, self.compute_backend)

        # Copy f_0 using backend-specific copy to f_1
        if self.compute_backend == ComputeBackend.JAX:
            f_1 = f_0.copy()
        else:
            wp.copy(f_1, f_0)

        # Process boundary conditions and update masks
        bc_mask, missing_mask = self._process_boundary_conditions(self.boundary_conditions, bc_mask, missing_mask)
        # Initialize auxiliary data if needed
        f_0, f_1 = self._initialize_auxiliary_data(self.boundary_conditions, f_0, f_1, bc_mask, missing_mask)

        return f_0, f_1, f_temp, bc_mask, missing_mask, U_0, u_num_displ_0, u_num_displ_1

    @classmethod
    def _process_boundary_conditions(cls, boundary_conditions, bc_mask, missing_mask):  # TODO: initialize the BCs, maybe OK
        """Process boundary conditions and update boundary masks."""
        # Check for boundary condition overlaps
        check_bc_overlaps(boundary_conditions, DefaultConfig.velocity_set.d, DefaultConfig.default_backend)
        # Create boundary maskers
        indices_masker = IndicesBoundaryMasker(
            velocity_set=DefaultConfig.velocity_set,
            precision_policy=DefaultConfig.default_precision_policy,
            compute_backend=DefaultConfig.default_backend,
        )
        # Split boundary conditions by type
        bc_with_vertices = [bc for bc in boundary_conditions if bc.mesh_vertices is not None]
        bc_with_indices = [bc for bc in boundary_conditions if bc.indices is not None]
        # Process indices-based boundary conditions
        if bc_with_indices:
            bc_mask, missing_mask = indices_masker(bc_with_indices, bc_mask, missing_mask)
        # Process mesh-based boundary conditions for 3D
        if DefaultConfig.velocity_set.d == 3 and bc_with_vertices:
            mesh_masker = MeshBoundaryMasker(
                velocity_set=DefaultConfig.velocity_set,
                precision_policy=DefaultConfig.default_precision_policy,
                compute_backend=DefaultConfig.default_backend,
            )
            for bc in bc_with_vertices:
                bc_mask, missing_mask = mesh_masker(bc, bc_mask, missing_mask)

        return bc_mask, missing_mask

    @staticmethod
    def _initialize_auxiliary_data(boundary_conditions, f_0, f_1, bc_mask, missing_mask):  # TODO: check when this is needed
        """Initialize auxiliary data for boundary conditions that require it."""
        for bc in boundary_conditions:
            if bc.needs_aux_init and not bc.is_initialized_with_aux_data:
                f_0, f_1 = bc.aux_data_init(f_0, f_1, bc_mask, missing_mask)
        return f_0, f_1

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,))
    def jax_implementation(self, f_0, f_1, bc_mask, missing_mask, omega, timestep):
        """
        Perform a single step of the lattice boltzmann method
        """
        # Cast to compute precision
        f_0 = self.precision_policy.cast_to_compute_jax(f_0)  # Untouched
        f_1 = self.precision_policy.cast_to_compute_jax(f_1)  # Untouched

        # Apply streaming, base streamer is periodic. TODO: Adjust for Dirichlet BCs.
        f_post_stream = self.stream_LED(f_0)  # Done

        # Apply boundary conditions.
        # Skipped for now, as bc = [] in sinewave_LED_file.
        for bc in self.boundary_conditions:
            if bc.implementation_step == ImplementationStep.STREAMING:
                f_post_stream = bc(
                    f_0,
                    f_post_stream,
                    bc_mask,
                    missing_mask,
                )

        # Compute the macroscopic variables, ie. the moments.
        # In LED we only need zeroth-order moment
        # In LED, we get v_num from U_num_tilde
        U_num_tilde = self.macroscopic_LED(f_post_stream)  # Done. TODO: add axternal forcing B_tilde

        # Compute equilibrium
        feq = self.equilibrium_LED(U_num_tilde)

        # Apply collision
        f_post_collision = self.collision_LED(f_post_stream, feq, U_num_tilde, omega)

        # Apply collision type boundary conditions
        for bc in self.boundary_conditions:
            f_post_collision = bc.update_bc_auxilary_data(f_post_stream, f_post_collision, bc_mask, missing_mask)
            if bc.implementation_step == ImplementationStep.COLLISION:
                f_post_collision = bc(
                    f_post_stream,
                    f_post_collision,
                    bc_mask,
                    missing_mask,
                )

        # Copy back to store precision
        f_1 = self.precision_policy.cast_to_store_jax(f_post_collision)

        return f_0, f_1

    def _construct_warp(self):
        # Set local constants
        _f_vec = wp.vec(self.velocity_set.q * 5, dtype=self.compute_dtype)
        _uxy_vec = wp.vec(2, dtype=self.compute_dtype)
        _U_num_tilde_vec = wp.vec(5, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(self.velocity_set.q * 5, dtype=wp.uint8)
        _opp_indices = self.velocity_set.opp_indices

        # Read the list of bc_to_id created upon instantiation
        bc_to_id = boundary_condition_registry.bc_to_id
        id_to_bc = boundary_condition_registry.id_to_bc

        # Gather IDs of ExtrapolationOutflowBC boundary conditions
        extrapolation_outflow_bc_ids = []
        for bc_name, bc_id in bc_to_id.items():
            if bc_name.startswith("ExtrapolationOutflowBC"):
                extrapolation_outflow_bc_ids.append(bc_id)
        # Group active boundary conditions
        active_bcs = set(boundary_condition_registry.id_to_bc[bc.id] for bc in self.boundary_conditions)

        _opp_indices = self.velocity_set.opp_indices

        @wp.func
        def apply_bc(
            index: Any,
            timestep: Any,
            _boundary_id: Any,
            missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
            is_post_streaming: bool,
        ):
            f_result = f_post

            # Unroll the loop over boundary conditions
            for i in range(wp.static(len(self.boundary_conditions))):
                if is_post_streaming:
                    if wp.static(self.boundary_conditions[i].implementation_step == ImplementationStep.STREAMING):
                        if _boundary_id == wp.static(self.boundary_conditions[i].id):
                            f_result = wp.static(self.boundary_conditions[i].warp_functional)(index, timestep, missing_mask, f_0, f_1, f_pre, f_post)
                else:
                    if wp.static(self.boundary_conditions[i].implementation_step == ImplementationStep.COLLISION):
                        if _boundary_id == wp.static(self.boundary_conditions[i].id):
                            f_result = wp.static(self.boundary_conditions[i].warp_functional)(index, timestep, missing_mask, f_0, f_1, f_pre, f_post)
                    if wp.static(self.boundary_conditions[i].id in extrapolation_outflow_bc_ids):
                        if _boundary_id == wp.static(self.boundary_conditions[i].id):
                            f_result = wp.static(self.boundary_conditions[i].update_bc_auxilary_data)(
                                index, timestep, missing_mask, f_0, f_1, f_pre, f_post
                            )
            return f_result

        @wp.func
        def get_thread_data(
            f0_buffer: wp.array4d(dtype=Any),
            f1_buffer: wp.array4d(dtype=Any),
            index: Any,
            uxy_buffer: wp.array4d(dtype=Any),
        ):
            # Read thread data for populations
            _f0_thread = _f_vec()
            _f1_thread = _f_vec()
            _uxy_thread = _uxy_vec()
            for l in range(20):
                # q-sized vector of pre-streaming populations
                _f0_thread[l] = self.compute_dtype(f0_buffer[l, index[0], index[1], index[2]])
                _f1_thread[l] = self.compute_dtype(f1_buffer[l, index[0], index[1], index[2]])
            for l in range(2):
                _uxy_thread[l] = self.compute_dtype(uxy_buffer[l, index[0], index[1], index[2]])

            return _f0_thread, _f1_thread, _uxy_thread

        # @wp.func
        # def apply_aux_recovery_bc(
        #     ind_feqex: Any,
        #     _boundary_id: Any,
        #     _missing_mask: Any,
        #     f_0: Any,
        #     _f1_thread: Any,
        # ):
        #     # Note:
        #     # In XLB, the BC auxiliary data (e.g. prescribed values of pressure or normal velocity) are stored in (i) central index of f_1 and/or
        #     # (ii) missing directions of f_1. Some BCs may or may not need all these available storage space. This function checks whether
        #     # the BC needs recovery of auxiliary data and then recovers the information for the next iteration (due to buffer swapping) by
        #     # writting the thread values of f_1 (i.e._f1_thread) into f_0.

        #     # Unroll the loop over boundary conditions
        #     for i in range(wp.static(len(self.boundary_conditions))):
        #         if wp.static(self.boundary_conditions[i].needs_aux_recovery):
        #             if _boundary_id == wp.static(self.boundary_conditions[i].id):
        #                 # Perform the swapping of data
        #                 # (i) Recover the values stored in the central index of f_1
        #                 f_0[0, index[0], index[1], index[2]] = self.store_dtype(_f1_thread[0])
        #                 # (ii) Recover the values stored in the missing directions of f_1
        #                 for l in range(1, self.velocity_set.q):
        #                     if _missing_mask[l] == wp.uint8(1):
        #                         f_0[_opp_indices[l], index[0], index[1], index[2]] = self.store_dtype(_f1_thread[_opp_indices[l]])

        @wp.kernel
        def kernel(
            f_0: wp.array4d(dtype=Any),
            f_star: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=Any),
            omega: Any,
            timestep: int,
            U_num_tilde: wp.array4d(dtype=Any),
            u_num_displ_0: wp.array4d(dtype=Any),
            u_num_displ_1: wp.array4d(dtype=Any),
        ):
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            
            # TODO: remove U_num_tilde from get thread
            _f0_thread, _fstar_thread, _uxy_thread = get_thread_data(f_0, f_star, index, u_num_displ_0)
            _f_post_stream = _f0_thread

            t = self.compute_dtype(timestep)* wp.delta_t_led

            # Collision
            # 1.a)
            _U_num_tilde = self.macroscopic_LED.warp_functional(_f_post_stream, index, t)

            # 1.b) - get displacement solution.
            if t > 0:  # This is also done in the matlab script..
                _u_num_displ = self.displacement_LED.warp_functional(_U_num_tilde, _uxy_thread)
            else:
                _u_num_displ = _uxy_thread
            
            # 1.c) Get local equilibrium populations
            _feq = self.equilibrium_LED.warp_functional(_U_num_tilde)
            # 1.d) Collision step
            _f_post_collision = self.collision_LED.warp_functional(_f_post_stream, _feq, omega)

            for l in range(5):
                U_num_tilde[l, index[0], index[1], index[2]] = self.store_dtype(_U_num_tilde[l])

            for l in range(20):
                f_star[l, index[0], index[1], index[2]] = self.store_dtype(_f_post_collision[l])

            for l in range(2):
                u_num_displ_1[l, index[0], index[1], index[2]] = self.store_dtype(_u_num_displ[l])

        return None, kernel

    # @Operator.register_backend(ComputeBackend.WARP)
    # def warp_implementation(self, f_0, f_1, bc_mask, missing_mask, omega, timestep):
    #     wp.launch(
    #         self.warp_kernel,
    #         inputs=[f_0, f_1, bc_mask, missing_mask, omega, timestep],
    #         dim=f_0.shape[1:],
    #     )
    #     return f_0, f_1
    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_0, f_star, bc_mask, omega, timestep, U_num_tilde, u_num_displ_0, u_num_displ_1):
        wp.launch(
            self.warp_kernel,
            inputs=[f_0, f_star, bc_mask, omega, timestep, U_num_tilde, u_num_displ_0, u_num_displ_1],
            dim=f_0.shape[1:],
        )

        return f_0, f_star, U_num_tilde, u_num_displ_1

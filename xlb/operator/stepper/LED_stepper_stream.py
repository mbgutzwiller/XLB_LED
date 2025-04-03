import warp as wp
from typing import Any

from xlb import DefaultConfig
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
from xlb.operator.stream import Stream_LED
from xlb.operator.collision import BGK_LED
from xlb.operator.equilibrium import Equilibrium_LED
from xlb.operator.macroscopic import Macroscopic_LED
from xlb.operator.displacement.displacement_LED import Displacement_LED
from xlb.operator.stepper import Stepper
from xlb.operator.boundary_condition.boundary_condition_LED import ImplementationStep_LED
from xlb.operator.boundary_condition.boundary_condition_registry import boundary_condition_registry
from xlb.operator.boundary_masker import IndicesBoundaryMasker, MeshBoundaryMasker
from xlb.helper import check_bc_overlaps
from xlb.helper.LED_solver_v2 import create_LED_fields
from xlb.helper.initializers import initialize_f_U_num_LED


class LinearElastodynamicsStepperStream(Stepper):
    def __init__(
        self,
        grid,
        boundary_conditions=[],
        collision_type="BGK_LED",
    ):
        super().__init__(grid, boundary_conditions)

        # Construct the collision operator, using equation (28) 
        if collision_type == "BGK_LED":
            self.collision_LED = BGK_LED(self.velocity_set, self.precision_policy, self.compute_backend)
        else:
            raise NotImplementedError(f"{collision_type} not implemented yet, but only BGK.")

        # Construct the operators
        self.stream_LED = Stream_LED(self.velocity_set, self.precision_policy, self.compute_backend)
        self.equilibrium_LED = Equilibrium_LED(self.velocity_set, self.precision_policy, self.compute_backend)
        self.macroscopic_LED = Macroscopic_LED(self.velocity_set, self.precision_policy, self.compute_backend)
        self.displacement_LED = Displacement_LED((self.velocity_set, self.precision_policy, self.compute_backend))

    def prepare_fields(self):
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
        _, f_0, f_1, missing_mask, bc_mask = create_LED_fields(
            grid=self.grid, compute_backend=self.compute_backend, precision_policy=self.precision_policy
        )

        f_0 , U_0, U_1, u_num_displ_0, u_num_displ_1 = initialize_f_U_num_LED(f_0, self.grid, self.precision_policy, self.compute_backend)

        # Copy f_0 using backend-specific copy to f_1
        if self.compute_backend == ComputeBackend.JAX:
            f_1 = f_0.copy()
        else:
            wp.copy(f_1, f_0)

        # Process boundary conditions and update masks
        bc_mask, missing_mask = self._process_boundary_conditions(self.boundary_conditions, bc_mask, missing_mask)
        # Initialize auxiliary data if needed
        f_0, f_1 = self._initialize_auxiliary_data(self.boundary_conditions, f_0, f_1, bc_mask, missing_mask)

        return f_0, f_1, bc_mask, missing_mask, U_0, U_1, u_num_displ_0, u_num_displ_1

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

    def _construct_warp(self):
        # Set local constants
        _f_vec = wp.vec(self.velocity_set.q * 5, dtype=self.compute_dtype)
        _uxy_vec = wp.vec(2, dtype=self.compute_dtype)
        _U_num_tilde_vec = wp.vec(5, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(self.velocity_set.q, dtype=wp.uint8)

        # Read the list of bc_to_id created upon instantiation
        bc_to_id = boundary_condition_registry.bc_to_id

        # Gather IDs of ExtrapolationOutflowBC boundary conditions
        extrapolation_outflow_bc_ids = []
        for bc_name, bc_id in bc_to_id.items():
            if bc_name.startswith("ExtrapolationOutflowBC"):
                extrapolation_outflow_bc_ids.append(bc_id)

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
                    if wp.static(self.boundary_conditions[i].implementation_step == ImplementationStep_LED.STREAMING):
                        if _boundary_id == wp.static(self.boundary_conditions[i].id):
                            f_result = wp.static(self.boundary_conditions[i].warp_functional)(index, timestep, missing_mask, f_0, f_1, f_pre, f_post)
                else:
                    if wp.static(self.boundary_conditions[i].implementation_step == ImplementationStep_LED.COLLISION):
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
            missing_mask: wp.array4d(dtype=Any),
            index: Any,
            uxy_buffer: wp.array4d(dtype=Any),
            U_num_tilde_buffer: wp.array4d(dtype=Any),
        ):
            # Read thread data for populations
            _f0_thread = _f_vec()
            _f1_thread = _f_vec()
            _uxy_thread = _uxy_vec()
            _U_num_tilde_thread = _U_num_tilde_vec()
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q * 5):
                # q-sized vector of pre-streaming populations
                _f0_thread[l] = self.compute_dtype(f0_buffer[l, index[0], index[1], index[2]])
                _f1_thread[l] = self.compute_dtype(f1_buffer[l, index[0], index[1], index[2]])
            for l in range(self.velocity_set.q):
                if missing_mask[l, index[0], index[1], index[2]]:
                    _missing_mask[l] = wp.uint8(1)
                else:
                    _missing_mask[l] = wp.uint8(0)
            for l in range(self.velocity_set.d):
                _uxy_thread[l] = self.compute_dtype(uxy_buffer[l, index[0], index[1], index[2]])
            for l in range(5):
                _U_num_tilde_thread[l] = self.compute_dtype(U_num_tilde_buffer[l, index[0], index[1], index[2]])

            return _f0_thread, _f1_thread, _missing_mask, _uxy_thread, _U_num_tilde_thread

        @wp.kernel
        def kernel(
            f_0: wp.array4d(dtype=Any),
            f_1: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=Any),
            missing_mask: wp.array4d(dtype=Any),
            omega: Any,
            timestep: int,
            U_num_tilde_1: wp.array4d(dtype=Any),
            u_num_displ_0: wp.array4d(dtype=Any),
            u_num_displ_1: wp.array4d(dtype=Any),
        ):
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            _boundary_id = bc_mask[0, index[0], index[1], index[2]]
            if _boundary_id == wp.uint8(255):
                return
            
            # Streaming
            # 2.a) stream on domain interior.
            _f_post_stream = self.stream_LED.warp_functional(f_0, index)

            _f0_thread, _f1_thread, _missing_mask, _uxy_thread, _U_num_tilde_thread = get_thread_data(f_0, f_1, missing_mask, index, u_num_displ_1, U_num_tilde_1)
            _f_post_collision = _f0_thread

            # 2.b) apply post streaming BCs.
            _f_post_stream = apply_bc(index, timestep, _boundary_id, _missing_mask, f_0, f_1, _f_post_collision, _f_post_stream, True)

            # 2.c) prepare displacement solution.
            _u_num_displ = self.displacement_LED.warp_functional(_U_num_tilde_thread, _uxy_thread)
            
            for l in range(self.velocity_set.q * 5):
                f_1[l, index[0], index[1], index[2]] = self.store_dtype(_f_post_stream[l])

            for l in range(self.velocity_set.d):
                u_num_displ_1[l, index[0], index[1], index[2]] = self.store_dtype(_u_num_displ[l])
            
        return None, kernel


    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_0, f_1, bc_mask, missing_mask, omega, timestep, U_num_tilde_1, u_num_displ_0, u_num_displ_1):
        wp.launch(
            self.warp_kernel,
            inputs=[f_0, f_1, bc_mask, missing_mask, omega, timestep, U_num_tilde_1, u_num_displ_0, u_num_displ_1],
            dim=f_0.shape[1:],
        )

        return f_0, f_1, u_num_displ_0, u_num_displ_1

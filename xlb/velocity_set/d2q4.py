# Description: Lattice class for 2D D2Q4 lattice.

import numpy as np

from xlb.velocity_set.velocity_set import VelocitySet


class D2Q4(VelocitySet):
    """
    Velocity Set for 2D D2Q4 lattice.

    D2Q4 stands for two-dimensional four-velocity model. It is the model used in the
    Lattice Boltzmann Method for simulating linear elastodynamics in two dimensions as Oliver et al developed it.
    """

    def __init__(self, precision_policy, compute_backend):
        # Construct the velocity vectors and weights
        # Make sure to match with equilibrium function and initialization.
        cx = [1, 0, -1, 0]
        cy = [0, 1, 0, -1]
        c = np.array(tuple(zip(cx, cy))).T
        print(c)
        w = np.array([1/4, 1/4, 1/4, 1/4])

        # Call the parent constructor
        super().__init__(2, 4, c, w, precision_policy=precision_policy, compute_backend=compute_backend)

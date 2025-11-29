import numpy as np

class Abstract:
    def __init__(self, X_bounds, U_bounds, W_bounds, cells_per_dim_x, cells_per_dim_u, Automaton_transit, Initial_state, Final_states):
        
        self.X_bounds = X_bounds
        self.W_bounds = W_bounds
        self.U_bounds = U_bounds
        
        self.dx_cell = (X_bounds[:, 1] - X_bounds[:, 0]) / cells_per_dim_x
        self.du_cell = (U_bounds[:, 1] - U_bounds[:, 0]) / cells_per_dim_u
        self.dw_cell = (W_bounds[:, 1] - W_bounds[:, 0])

        self.M_x = self.build_multiplier_array(cells_per_dim_x)
        self.M_u = self.build_multiplier_array(cells_per_dim_u)
        
        self.N_x = self.M_x[-1]
        self.N_u = self.M_u[-1]

    def idx_to_coord(self, state_idx):
        """
        Convert a discrete state index to its cell coordinates (i, j, k, ...).
        
        Uses the multiplier array to decompose a linear state index into multi-dimensional coordinates.
        This is the inverse of coord_to_idx operation.
        
        Args:
            state_idx: Linear index of the discrete state
            
        Returns:
            Cell coordinates as array
        """
        state_coord = np.zeros(len(self.M_x) - 1, dtype=int)
        for d in range(len(self.M_x) - 1):
            remainder = (state_idx - 1) % self.M_x[d + 1]
            state_coord[d] = np.floor(remainder / self.M_x[d]) + 1

        return state_coord

    def coord_to_idx(self, state_coord):
        """
        Convert cell coordinates (i, j, k, ...) to a discrete state index.
        
        This is the inverse of idx_to_coord operation. Computes the linear state index
        from multi-dimensional cell coordinates using the multiplier array.
        
        Args:
            state_coord: Multi-dimensional cell coordinates
            
        Returns:
            Linear discrete state index
        """
        return (state_coord - 1).transpose() @ self.M_x[0: len(state_coord)] + 1

    def discretize_control(self):
        """
        Generate the set of all discrete control inputs.
        
        Creates the control input space by discretizing the continuous control bounds
        using the provided discretization width and multiplier array.
        
        Returns:
            U_disc: Matrix of shape (control_dimension, number_of_control_inputs)
                   containing all discrete control values
        """
        U_disc = np.zeros((len(self.U_bounds), self.N_u))
        for control_idx in range(1, self.N_u + 1):
            U_disc[:, control_idx - 1] = (
                self.U_bounds[:, 0] + 
                (self.idx_to_coord(control_idx) - 1) * self.du_cell
            )
        
        return U_disc

    def build_multiplier_array(cells_per_dim):
        """
        Automatically construct multiplier array from cells per dimension.
        
        Converts array of cell counts per dimension into the multiplier array
        used for efficient index-coordinate conversion.
        
        Args:
            cells_per_dim: Array of cell counts [K_1, K_2, ..., K_n]
            
        Returns:
            M: Multiplier array [M_0, M_1, ..., M_n] where M_k = product of K_i
        """
        M = np.ones(len(cells_per_dim) + 1, dtype=int)
        for k in range(len(cells_per_dim)):
            M[k + 1] = M[k] * cells_per_dim[k]
        return M


    def state2coord(state_idx, p_x):
        pass


    def coordbox2state(cMin, cMax, grid_xi):
        pass

    def labelStates():
        pass

    def h1(psi, xi, labelingXi):
        pass
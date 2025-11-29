import numpy as np

class Abstract:
    def __init__(self, X_bounds, U_bounds, W_bounds, cells_per_dim_x, cells_per_dim_u, Automaton_transit, Initial_state, Final_states, angular_dims_x=None):
        
        self.X_bounds = X_bounds
        self.W_bounds = W_bounds
        self.U_bounds = U_bounds
        self.angular_dims_x = angular_dims_x or []  # 0-indexed dimensions that are angular (e.g., [2] for 3rd dimension)
        
        self.dx_cell = (X_bounds[:, 1] - X_bounds[:, 0]) / cells_per_dim_x
        self.du_cell = (U_bounds[:, 1] - U_bounds[:, 0]) / cells_per_dim_u
        self.dw_cell = (W_bounds[:, 1] - W_bounds[:, 0])

        self.M_x = self.build_multiplier_array(cells_per_dim_x)
        self.M_u = self.build_multiplier_array(cells_per_dim_u)
        
        self.N_x = self.M_x[-1]
        self.N_u = self.M_u[-1]

    def continuous_to_cell_idx(self, continuous_coord):
        """
        Convert continuous state coordinates (within X_bounds) to cell indices.
        
        Maps a continuous point in the state space to its corresponding cell index
        in the discretized grid.
        
        Args:
            continuous_coord: Continuous state coordinates within X_bounds
            
        Returns:
            Cell indices as array (0-indexed)
        """
        cell_indices = np.floor((continuous_coord - self.X_bounds[:, 0]) / self.dx_cell).astype(int)
        return np.clip(cell_indices, 0, self.M_x[1:] - 1)

    def continuous_control_to_cell_idx(self, continuous_control):
        """
        Convert continuous control inputs (within U_bounds) to cell indices.
        
        Maps a continuous control point in the control space to its corresponding 
        cell index in the discretized grid.
        
        Args:
            continuous_control: Continuous control inputs within U_bounds
            
        Returns:
            Control cell indices as array (0-indexed)
        """
        cell_indices = np.floor((continuous_control - self.U_bounds[:, 0]) / self.du_cell).astype(int)
        return np.clip(cell_indices, 0, self.M_u[1:] - 1)

    def idx_to_coord(self, state_idx):
        """
        Convert a discrete state index to its cell coordinates (i, j, k, ...).
        
        Uses the multiplier array to decompose a linear state index into multi-dimensional 
        cell coordinates (0-indexed). This is the inverse of coord_to_idx operation.
        
        Args:
            state_idx: Linear index of the discrete state (0-indexed)
            
        Returns:
            Cell coordinates as array (0-indexed)
        """
        state_coord = np.zeros(len(self.M_x) - 1, dtype=int)
        for d in range(len(self.M_x) - 1):
            remainder = state_idx % self.M_x[d + 1]
            state_coord[d] = remainder // self.M_x[d]

        return state_coord

    def coord_to_idx(self, cell_coord):
        """
        Convert cell coordinates (i, j, k, ...) to a discrete state index.
        
        This is the inverse of idx_to_coord operation. Computes the linear state index
        from multi-dimensional cell coordinates using the multiplier array.
        
        Note: cell_coord should be cell indices (0-indexed), not continuous values.
        Use continuous_to_cell_idx() to convert continuous coordinates to cell indices first.
        
        Args:
            cell_coord: Multi-dimensional cell coordinates (0-indexed)
            
        Returns:
            Linear discrete state index (0-indexed)
        """
        return cell_coord @ self.M_x[0: len(cell_coord)]

    def control_idx_to_coord(self, control_idx):
        """
        Convert a discrete control index to its cell coordinates (i, j, k, ...).
        
        Similar to idx_to_coord but uses the control multiplier array (M_u)
        instead of the state multiplier array.
        
        Args:
            control_idx: Linear index of the discrete control (0-indexed)
            
        Returns:
            Control cell coordinates as array (0-indexed)
        """
        control_coord = np.zeros(len(self.M_u) - 1, dtype=int)
        for d in range(len(self.M_u) - 1):
            remainder = control_idx % self.M_u[d + 1]
            control_coord[d] = remainder // self.M_u[d]

        return control_coord

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
        for control_idx in range(self.N_u):
            U_disc[:, control_idx] = (
                self.U_bounds[:, 0] + 
                (self.control_idx_to_coord(control_idx)) * self.du_cell
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
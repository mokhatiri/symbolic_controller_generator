import numpy as np

class Discretisation:
    def __init__(self, X_bounds, U_bounds, W_bounds, cells_per_dim_x, cells_per_dim_u, angular_dims_x=None):
        
        self.X_bounds = X_bounds
        self.W_bounds = W_bounds
        self.U_bounds = U_bounds
        self.angular_dims_x = angular_dims_x or []  # 0-indexed dimensions that are angular (e.g., [2] for 3rd dimension)
        
        self.dx_cell = (X_bounds[:, 1] - X_bounds[:, 0]) / cells_per_dim_x
        self.du_cell = (U_bounds[:, 1] - U_bounds[:, 0]) / cells_per_dim_u
        self.dw_cell = (W_bounds[:, 1] - W_bounds[:, 0])
        self.W_width = W_bounds[:, 1] - W_bounds[:, 0]  # Width of each disturbance dimension

        self.M_x = self.build_multiplier_array(cells_per_dim_x)
        self.M_u = self.build_multiplier_array(cells_per_dim_u)
        
        self.N_x = self.M_x[-1]
        self.N_u = self.M_u[-1]

    def continuous_to_discrete(self, continuous_values, bounds, cell_sizes, multiplier_array):
        """
        General-purpose method to convert continuous values to discrete indices.
        
        This is a generalized version of continuous_to_cell_idx and continuous_control_to_cell_idx.
        Maps a continuous point in a bounded space to its corresponding cell index in the discretized grid.
        
        Args:
            continuous_values: Continuous coordinates within bounds
            bounds: [min, max] bounds for each dimension
            cell_sizes: Cell width for each dimension (e.g., dx_cell or du_cell)
            multiplier_array: Multiplier array for this space (e.g., M_x or M_u)
            
        Returns:
            Discrete cell indices as array (0-indexed), clipped to valid range
        """
        cell_indices = np.floor((continuous_values - bounds[:, 0]) / cell_sizes).astype(int)
        # Clip to valid range [0, cells_per_dim - 1]
        max_indices = multiplier_array[1:] - 1
        return np.clip(cell_indices, 0, max_indices)

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
        return self.continuous_to_discrete(continuous_coord, self.X_bounds, self.dx_cell, self.M_x)

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
        return self.continuous_to_discrete(continuous_control, self.U_bounds, self.du_cell, self.M_u)

    def idx_to_continuous(self, state_idx, bounds, cell_sizes):
        """
        Convert a discrete index to continuous coordinates at cell center.
        
        General-purpose method to get the continuous center point of a discretized cell.
        
        Args:
            state_idx: Linear discrete index (0-indexed)
            bounds: [min, max] bounds for the space
            cell_sizes: Cell width for each dimension
            
        Returns:
            Continuous coordinates at the cell center
        """
        state_coord = self.idx_to_coord(state_idx)
        return bounds[:, 0] + (state_coord + 0.5) * cell_sizes

    def discretize_state(self):
        """
        Generate the set of all discrete state space centers.
        
        Returns the continuous center point of each discretized state cell.
        Similar to discretize_control but for states.
        
        Returns:
            X_disc: Matrix of shape (state_dimension, number_of_states)
                   containing continuous state values at cell centers
        """
        X_disc = np.zeros((len(self.X_bounds), self.N_x))
        for state_idx in range(self.N_x):
            X_disc[:, state_idx] = self.idx_to_continuous(state_idx, self.X_bounds, self.dx_cell)
        return X_disc

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
        using the provided discretization width. Returns the continuous center point
        of each discretized cell.
        
        Returns:
            U_disc: Matrix of shape (control_dimension, number_of_control_inputs)
                   containing continuous control values at cell centers
        """
        U_disc = np.zeros((len(self.U_bounds), self.N_u))
        for control_idx in range(self.N_u):
            # Convert discrete control index to cell coordinates (0-indexed)
            control_coord = self.control_idx_to_coord(control_idx)
            # Compute continuous control at cell center
            U_disc[:, control_idx] = (
                self.U_bounds[:, 0] + 
                (control_coord + 0.5) * self.du_cell
            )
        
        return U_disc

    @staticmethod
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
import numpy as np
import pandas as pd

class SymbolicAbstraction:
    """
    Computes the discrete symbolic abstraction of a continuous nonlinear system.
    
    This class implements the abstraction procedure that maps a continuous dynamical system
    to a finite symbolic model, enabling discrete controller synthesis.
    """

    def __init__(self, System, Discretisation):
        """
        Initialize the symbolic abstraction generator.
        Args:
            System: Instance of the continuous system dynamics
            Discretisation: Instance of the discretisation parameters
            T: Transition system (to be computed)
        """
        self.System = System
        self.Discretisation = Discretisation
        self.T = self.compute_symbolic_model()

    def compute_symbolic_model(self):
        """
        Compute the discrete symbolic model (transition system).
        
        For each discrete state and control input pair (state_idx, control_idx), this method computes
        the set of reachable successor states by:
        1. Computing the continuous center point of the cell
        2. Applying one step of the continuous dynamics
        3. Computing the reachable set interval using interval arithmetic
        4. Mapping this reachable set back to discrete cells
        5. Handling wraparound for angular states (theta coordinate)
        
        Returns:
            T: 3D array of shape (N_x, N_u, 2)
               where T[state_idx, control_idx, :] = [min_successor, max_successor] (successor state range)
        """
        
        # Disturbance at equilibrium point (center of disturbance bounds)
        w_center = 0.5 * (self.Discretisation.W_bounds[:, 0] + self.Discretisation.W_bounds[:, 1])
        
        # Initialize transition table of the automate
        T = np.zeros((self.Discretisation.N_x, self.Discretisation.N_u, 2), dtype=int)
        trans_count = 0
        
        # Precompute discretized controls
        U_disc = self.Discretisation.discretize_control()
        
        for state_idx in range(1, self.Discretisation.N_x + 1):
            # Convert state index to cell coordinates
            state_coord = self.Discretisation.idx_to_coord(state_idx)
            # Compute continuous center of the state cell
            x_center = self.Discretisation.X_bounds[:, 0] + (state_coord - 0.5) * self.Discretisation.dx_cell
            
            for control_idx in range(1, self.Discretisation.N_u + 1):
                # Get the discrete control value
                u_control = U_disc[:, control_idx - 1]

                # Compute successor state center using continuous dynamics
                x_succ_center = self.System.f(x_center, u_control, w_center)

                # Compute reachable set bounds using Jacobian-based interval arithmetic
                # dx_succ = 0.5 * |J_x| @ dx + 0.5 * |J_w| @ dw
                dx_succ = (0.5 * np.abs(self.System.Jx(u_control)) @ self.Discretisation.dx_cell + 0.5 * np.abs(self.System.Jw(u_control)) @ self.Discretisation.W_width)
                
                # Reachable set interval [x_min, x_max]
                R = np.vstack([
                    x_succ_center - dx_succ, 
                    x_succ_center + dx_succ
                ])
                
                # Handle angular wraparound for the third dimension (theta)
                # This accounts for the circular nature of angles in [-pi, pi]
                if R[0, 2] < -np.pi and R[1, 2] >= -np.pi:
                    R[0, 2] += 2 * np.pi
                elif R[0, 2] < -np.pi and R[1, 2] < -np.pi:
                    R[0, 2] += 2 * np.pi
                    R[1, 2] += 2 * np.pi
                elif R[1, 2] > np.pi and R[0, 2] <= np.pi:
                    R[1, 2] -= 2 * np.pi
                elif R[1, 2] > np.pi and R[0, 2] > np.pi:
                    R[1, 2] -= 2 * np.pi
                    R[0, 2] -= 2 * np.pi

                R = R.transpose()
                
                # Check if reachable set is within state bounds and map to discrete cells
                if (np.all(R[:, 0] >= self.Discretisation.X_bounds[:, 0]) and 
                    np.all(R[:, 1] <= self.Discretisation.X_bounds[:, 1])):
                    
                    # Compute minimum and maximum successor cells
                    min_succ_coord = np.floor(
                        (R[:, 0] - self.Discretisation.X_bounds[:, 0]) / self.Discretisation.dx_cell
                    ).astype(int) + 1
                    max_succ_coord = np.ceil(
                        (R[:, 1] - self.Discretisation.X_bounds[:, 0]) / self.Discretisation.dx_cell
                    ).astype(int)
                    
                    min_successor = self.Discretisation.coord_to_idx(min_succ_coord)
                    max_successor = self.Discretisation.coord_to_idx(max_succ_coord)
                    
                    T[state_idx - 1, control_idx - 1] = [min_successor, max_successor]
                    trans_count += 1
        
        return T
    

    """
        ---- Helper Methods ----
    """


    def __getitem__(self, key):
        """Enable indexing notation for accessing symbolic model."""
        return self.T[key]

    def save_symbolic_model(self, fname):
        """
        Save the symbolic model to a CSV file for persistent storage and reuse.
        
        Args:
            fname: Name of the CSV file to save to (stored in Models/ directory)
        """
        rows = []
        
        for state_idx in range(self.Discretisation.N_x):
            for control_idx in range(self.Discretisation.N_u):

                min_succ = self.T[state_idx, control_idx, 0]
                max_succ = self.T[state_idx, control_idx, 1]
                

                rows.append([state_idx + 1, control_idx + 1, min_succ, max_succ])
        
        df = pd.DataFrame(rows, columns=['State Index', 'Input Index', 'Min Successor', 'Max Successor'])
        df.to_csv("Models/"+fname, index=False)

    def load_symbolic_model(self, fname="symbolic_model.csv"):
        """
        Load a previously computed symbolic model from a CSV file.
        
        Args:
            fname: Name of the CSV file to load from (in Models/ directory)
        """
        df = pd.read_csv("Models/"+fname)
        
        T = np.zeros((self.Discretisation.N_x, self.Discretisation.N_u, 2), dtype=int)
        
        for _, row in df.iterrows():
            state_idx = int(row['State Index']) - 1 
            control_idx = int(row['Input Index']) - 1
            min_succ = int(row['Min Successor'])
            max_succ = int(row['Max Successor'])
            
            T[state_idx, control_idx, 0] = min_succ
            T[state_idx, control_idx, 1] = max_succ
        
        self.T = T
        return self
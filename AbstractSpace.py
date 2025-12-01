import numpy as np
import pandas as pd
import os

class AbstractSpace:
    """
    Computes the discrete symbolic abstraction of a continuous nonlinear system.
    
    This class implements the abstraction procedure that maps a continuous dynamical system
    to a finite symbolic model, enabling discrete controller synthesis.
    """

    def __init__(self, System, Discretisation, model_dir='./Models'):
        """
        Initialize the symbolic abstraction generator.
        Args:
            System: Instance of the continuous system dynamics
            Discretisation: Instance of the discretisation parameters
            model_dir: Directory to save/load model files (default: './Models')
        """
        self.System = System
        self.Discretisation = Discretisation
        self.model_dir = model_dir
        
        # check if a symbolic model has already been computed and saved
        try:
            print("Attempting to load existing symbolic model from file...")
            self.transition = self.load_symbolic_model("symbolic_model.csv").transition
            print("✓ Loaded existing symbolic model from file.")
        except FileNotFoundError:
            print("✗ No existing symbolic model found. Computing new symbolic model...")
            self.transition = self.compute_symbolic_model()

        # save the symbolic model for future use
        self.save_symbolic_model("symbolic_model.csv")

    def normalize_angular_bounds(self, R):
        """
        Normalize angular dimensions to [-pi, pi].
        
        For each dimension marked as angular in the discretisation,
        this method wraps the reachable set bounds to the [-pi, pi] range.
        
        Args:
            R: Reachable set bounds array of shape (2, n_dims) with [min_coords; max_coords]
        
        Returns:
            R with angular dimensions normalized to [-pi, pi]
        """
        for dim in self.Discretisation.angular_dims_x:
            # Normalize both min and max to [-pi, pi]
            R[0, dim] = np.mod(R[0, dim] + np.pi, 2 * np.pi) - np.pi
            R[1, dim] = np.mod(R[1, dim] + np.pi, 2 * np.pi) - np.pi
        
        return R

    def map_continuous_to_discrete_cells(self, R):
        """
        Map continuous reachable set bounds to discrete cell coordinates.
        
        Generalizes the repeated computation of mapping continuous intervals to discrete cells.
        Handles the floor/ceil logic for interval discretization.
        
        Args:
            R: Reachable set with shape (n_dims, 2) where [:, 0] = min, [:, 1] = max
            
        Returns:
            Tuple of (min_successor_idx, max_successor_idx) as discrete state indices
        """
        # Convert continuous bounds to cell coordinates
        min_succ_coord = np.floor(
            (R[:, 0] - self.Discretisation.X_bounds[:, 0]) / self.Discretisation.dx_cell
        ).astype(int)
        max_succ_coord = np.ceil(
            (R[:, 1] - self.Discretisation.X_bounds[:, 0]) / self.Discretisation.dx_cell
        ).astype(int) - 1
        
        # Convert cell coordinates to discrete state indices
        min_successor = self.Discretisation.coord_to_idx(min_succ_coord)
        max_successor = self.Discretisation.coord_to_idx(max_succ_coord)
        
        return min_successor, max_successor

    def compute_reachable_set(self, x_center, u_control, w_center):
        """
        Compute reachable set bounds from a single point using Jacobian-based interval arithmetic.
        
        Generalizes the reachable set computation for code reusability.
        
        Args:
            x_center: Continuous state at cell center
            u_control: Continuous control input
            w_center: Continuous disturbance at center
            
        Returns:
            R: Reachable set bounds array of shape (2, n_dims) with [min; max]
        """
        # Compute successor state center using continuous dynamics
        x_succ_center = self.System.f(x_center, u_control, w_center)
        
        # Compute uncertainty bounds using Jacobian-based interval arithmetic
        # Uncertainty from discretization: 0.5 * dx_cell
        # Uncertainty from disturbance: 0.5 * W_width
        dx_succ = (0.5 * np.abs(self.System.Jx(u_control)) @ self.Discretisation.dx_cell +
                   0.5 * np.abs(self.System.Jw(u_control)) @ self.Discretisation.W_width)
        
        # Reachable set interval [x_min, x_max]
        R = np.vstack([
            x_succ_center - dx_succ,
            x_succ_center + dx_succ
        ])
        
        return R

    def compute_symbolic_model(self):
        """
        Compute the discrete symbolic model (transition system).
        
        For each discrete state and control input pair (state_idx, control_idx), this method computes
        the set of reachable successor states by:
        1. Computing the continuous center point of the cell
        2. Applying one step of the continuous dynamics
        3. Computing the reachable set interval using interval arithmetic
        4. Mapping this reachable set back to discrete cells
        5. Normalizing angular dimensions to [-pi, pi]
        
        Returns:
            T: 3D array of shape (N_x, N_u, 2)
               where T[state_idx, control_idx, :] = [min_successor, max_successor] (successor state range)
        """
        
        # Disturbance at equilibrium point (center of disturbance bounds)
        w_center = 0.5 * (self.Discretisation.W_bounds[:, 0] + self.Discretisation.W_bounds[:, 1])
        
        # Initialize transition table
        T = np.zeros((self.Discretisation.N_x, self.Discretisation.N_u, 2), dtype=int)
        
        # Precompute discretized controls
        U_disc = self.Discretisation.discretize_control()
        
        for state_idx in range(self.Discretisation.N_x):
            # Get continuous state at cell center
            x_center = self.Discretisation.idx_to_continuous(state_idx, self.Discretisation.X_bounds, self.Discretisation.dx_cell)
            
            for control_idx in range(self.Discretisation.N_u):
                # Get the continuous control value (already at cell center from discretize_control)
                u_control = U_disc[:, control_idx]

                # Compute reachable set bounds using Jacobian-based interval arithmetic
                R = self.compute_reachable_set(x_center, u_control, w_center)
                
                # Normalize angular dimensions to [-pi, pi]
                R = self.normalize_angular_bounds(R)

                R = R.transpose()
                
                # Check if reachable set is within state bounds and map to discrete cells
                if (np.all(R[:, 0] >= self.Discretisation.X_bounds[:, 0]) and 
                    np.all(R[:, 1] <= self.Discretisation.X_bounds[:, 1])):
                    
                    # Map continuous reachable set back to discrete cell indices
                    min_successor, max_successor = self.map_continuous_to_discrete_cells(R)
                    
                    T[state_idx, control_idx] = [min_successor, max_successor]
        

        return T
    

    """
        ---- Helper Methods ----
    """


    def __getitem__(self, key):
        """Enable indexing notation for accessing symbolic model."""
        return self.transition[key]

    def save_symbolic_model(self, fname):
        """
        Save the symbolic model to a CSV file for persistent storage and reuse.
        
        Args:
            fname: Name of the CSV file to save to (stored in model_dir)
        """
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        rows = []
        
        for state_idx in range(self.Discretisation.N_x):
            for control_idx in range(self.Discretisation.N_u):

                min_succ = self.transition[state_idx, control_idx, 0]
                max_succ = self.transition[state_idx, control_idx, 1]
                

                rows.append([state_idx, control_idx, min_succ, max_succ])
        
        df = pd.DataFrame(rows, columns=['State Index', 'Input Index', 'Min Successor', 'Max Successor'])
        df.to_csv(os.path.join(self.model_dir, fname), index=False)

    def load_symbolic_model(self, fname="symbolic_model.csv"):
        """
        Load a previously computed symbolic model from a CSV file.
        
        Args:
            fname: Name of the CSV file to load from (in model_dir)
        
        Raises:
            FileNotFoundError: If the model file doesn't exist
        """
        model_path = os.path.join(self.model_dir, fname)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Symbolic model file not found: {model_path}")
        
        df = pd.read_csv(model_path)
        
        T = np.zeros((self.Discretisation.N_x, self.Discretisation.N_u, 2), dtype=int)
        
        for _, row in df.iterrows():
            state_idx = int(row['State Index'])
            control_idx = int(row['Input Index'])
            min_succ = int(row['Min Successor'])
            max_succ = int(row['Max Successor'])
            
            T[state_idx, control_idx, 0] = min_succ
            T[state_idx, control_idx, 1] = max_succ
        
        self.transition = T
        return self
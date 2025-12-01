import numpy as np
import pandas as pd
import os
import json
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

class AbstractSpace:
    """
    Computes the discrete symbolic abstraction of a continuous nonlinear system.
    
    This class implements the abstraction procedure that maps a continuous dynamical system
    to a finite symbolic model, enabling discrete controller synthesis.
    """

    def __init__(self, System, Discretisation, model_dir='./Models', num_threads=None):
        """
        Initialize the symbolic abstraction generator.
        Args:
            System: Instance of the continuous system dynamics
            Discretisation: Instance of the discretisation parameters
            model_dir: Directory to save/load symbolic models (default: './Models')
            num_threads: Number of CPU threads for parallel computation (default: None = auto-detect)
            T: Transition system (to be computed)
        """
        self.System = System
        self.Discretisation = Discretisation
        self.model_dir = model_dir
        self.num_threads = num_threads or os.cpu_count() or 4
        self._inverse_transition = None  # Will store the main transition map
        
        # check if a symbolic model has already been computed and saved
        try:
            print("Attempting to load existing symbolic model (inverse map) from file...")
            self.load_symbolic_model("inverse_transition.csv")
            print("✓ Loaded existing symbolic model from file.")
        except FileNotFoundError:
            print("✗ No existing symbolic model found. Computing new symbolic model...")
            print(f"  Using {self.num_threads} CPU threads for parallel computation...")
            self._inverse_transition = self.compute_inverse_symbolic_model()

        # save the symbolic model for future use
        self.save_symbolic_model("inverse_transition.csv")
    
    @property
    def inverse_transition(self):
        """Return the inverse transition map (primary storage)."""
        return self._inverse_transition
    
    @inverse_transition.setter
    def inverse_transition(self, value):
        """Allow setting inverse transition map."""
        self._inverse_transition = value
    
    def get_successors(self, state_idx, control_idx):
        """
        Get successor states for a given (state, control) pair from inverse map.
        
        This replaces the forward transition lookup by searching the inverse map.
        
        Args:
            state_idx: State index
            control_idx: Control index
            
        Returns:
            List of successor state indices
        """
        successors = []
        # Search inverse map for all states that have (state_idx, control_idx) as predecessor
        for succ_state, predecessors in self._inverse_transition.items():
            for pred_state, pred_control in predecessors:
                if pred_state == state_idx and pred_control == control_idx:
                    successors.append(succ_state)
                    break
        return successors

    def normalize_angular_bounds(self, R):
        """
        Don't normalize angular bounds - preserve intervals for proper discretization.
        
        The old working code didn't normalize angular dimensions before discretization.
        Angular wrapping is handled during the discretization mapping instead.
        
        Args:
            R: Reachable set bounds array of shape (2, n_dims) with [min_coords; max_coords]
        
        Returns:
            R (unchanged - intervals preserved)
        """
        # Don't normalize! Angular wrapping is handled during discretization.
        return R

    def map_continuous_to_discrete_cells(self, R):
        """
        Map continuous reachable set bounds to discrete cell coordinates.
        
        Handles angular dimensions with wrapping (modulo arithmetic).
        This matches the behavior of the old working code.
        
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
        
        # Get the number of cells per dimension
        cells_per_dim = np.diff(self.Discretisation.M_x)
        
        # Handle angular dimensions with wrapping (modulo arithmetic)
        for dim in self.Discretisation.angular_dims_x:
            # Wrap coordinates to valid range [0, cells_per_dim-1]
            min_succ_coord[dim] = min_succ_coord[dim] % cells_per_dim[dim]
            max_succ_coord[dim] = max_succ_coord[dim] % cells_per_dim[dim]
        
        # Clamp non-angular dimensions to valid range
        for dim in range(len(min_succ_coord)):
            if dim not in self.Discretisation.angular_dims_x:
                min_succ_coord[dim] = np.clip(min_succ_coord[dim], 0, cells_per_dim[dim] - 1)
                max_succ_coord[dim] = np.clip(max_succ_coord[dim], 0, cells_per_dim[dim] - 1)
        
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

    def compute_inverse_symbolic_model(self):
        """
        Compute the inverse transition map directly (memory-efficient approach).
        
        Instead of storing a dense forward transition table, we build a sparse inverse map:
        inverse_map[successor_state] = [(predecessor_state, control), ...]
        
        This is more memory-efficient since most state-control pairs have few successors.
        
        Returns:
            inverse_map: Dictionary mapping successor states to list of (predecessor, control) tuples
        """
        from collections import defaultdict
        
        print("  Computing inverse transition map directly (memory-efficient)...")
        
        # Disturbance at equilibrium point (center of disturbance bounds)
        w_center = 0.5 * (self.Discretisation.W_bounds[:, 0] + self.Discretisation.W_bounds[:, 1])
        
        # Initialize inverse map as defaultdict
        inverse_map = defaultdict(list)
        
        # Precompute discretized controls
        U_disc = self.Discretisation.discretize_control()
        
        total_transitions = 0
        
        for state_idx in range(self.Discretisation.N_x):
            # Get continuous state at cell center
            x_center = self.Discretisation.idx_to_continuous(state_idx, self.Discretisation.X_bounds, self.Discretisation.dx_cell)
            
            for control_idx in range(self.Discretisation.N_u):
                # Get the continuous control value (already at cell center from discretize_control)
                u_control = U_disc[:, control_idx]

                # Compute reachable set bounds using Jacobian-based interval arithmetic
                R = self.compute_reachable_set(x_center, u_control, w_center)
                
                R = R.transpose()
                
                # Check if reachable set is within state bounds (non-angular dimensions only)
                valid = True
                for dim in range(R.shape[0]):
                    if dim not in self.Discretisation.angular_dims_x:
                        if R[dim, 0] < self.Discretisation.X_bounds[dim, 0] or R[dim, 1] > self.Discretisation.X_bounds[dim, 1]:
                            valid = False
                            break
                
                if valid:
                    # Map continuous reachable set back to discrete cell indices
                    min_successor, max_successor = self.map_continuous_to_discrete_cells(R)
                    
                    # Add all successors to inverse map
                    # Handle wrapped intervals (when min > max for angular dimensions)
                    if min_successor <= max_successor:
                        # Normal contiguous range
                        for succ_idx in range(min_successor, max_successor + 1):
                            inverse_map[succ_idx].append((state_idx, control_idx))
                            total_transitions += 1
                    else:
                        # Wrapped range: [min, N_x-1] and [0, max]
                        for succ_idx in range(min_successor, self.Discretisation.N_x):
                            inverse_map[succ_idx].append((state_idx, control_idx))
                            total_transitions += 1
                        for succ_idx in range(0, max_successor + 1):
                            inverse_map[succ_idx].append((state_idx, control_idx))
                            total_transitions += 1
            
            # Progress logging and memory cleanup
            if (state_idx + 1) % max(1, 1000 // max(1, self.Discretisation.N_u)) == 0:
                progress = 100 * (state_idx + 1) / self.Discretisation.N_x
                print(f"    Progress: {progress:.1f}% ({state_idx + 1}/{self.Discretisation.N_x} states, {total_transitions} transitions)")
                gc.collect()
        
        print(f"  ✓ Computed inverse map: {len(inverse_map)} states with transitions, {total_transitions} total transitions")
        return dict(inverse_map)
    

    """
        ---- Helper Methods ----
    """

    def __getitem__(self, key):
        """
        Enable indexing notation for accessing successors.
        Note: This now works differently - returns list of successors instead of range.
        
        Usage: AbstractSpace[state_idx, control_idx] returns list of successor states
        """
        if isinstance(key, tuple) and len(key) == 2:
            state_idx, control_idx = key
            return self.get_successors(state_idx, control_idx)
        else:
            raise KeyError("Expected tuple of (state_idx, control_idx)")

    def save_symbolic_model(self, fname):
        """
        Save the inverse transition map to CSV file for persistent storage and reuse.
        
        Args:
            fname: Name of the CSV file to save to (stored in model_dir)
        """
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save inverse transition map (sparse format)
        rows = []
        for successor_state, predecessors in self._inverse_transition.items():
            for predecessor_state, control_idx in predecessors:
                rows.append([predecessor_state, control_idx, successor_state])
        
        df = pd.DataFrame(rows, columns=['Predecessor State', 'Control Index', 'Successor State'])
        df.to_csv(f"{self.model_dir}/{fname}", index=False)
        print(f"  ✓ Saved {len(rows)} transitions to {self.model_dir}/{fname}")

    def load_symbolic_model(self, fname="inverse_transition.csv"):
        """
        Load the inverse transition map from CSV file.
        
        Args:
            fname: Name of the CSV file to load from (in model_dir)
        """
        from collections import defaultdict
        
        # Load inverse transition map
        df = pd.read_csv(f"{self.model_dir}/{fname}")
        
        inverse_map = defaultdict(list)
        for _, row in df.iterrows():
            predecessor_state = int(row['Predecessor State'])
            control_idx = int(row['Control Index'])
            successor_state = int(row['Successor State'])
            
            inverse_map[successor_state].append((predecessor_state, control_idx))
        
        self._inverse_transition = dict(inverse_map)
        print(f"  ✓ Loaded {len(df)} transitions from {self.model_dir}/{fname}")
        
        return self
    

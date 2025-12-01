import numpy as np
import time

class ControllerSynthesis:

    def __init__(self, Automaton, model_dir='./Models'):
        """
        Initialize the controller synthesis engine.
        
        Args:
            Automaton: Instance of the product automaton (system × specification)
            model_dir: Directory to save/load model files (default: './Models')
        """
        self.Automaton = Automaton
        self.model_dir = model_dir
        self.V = None
        self.h = None
        
        # Build mapping from spec state indices to names
        # Extract all unique spec states from the transition dictionary keys
        spec_states = set()
        for (state, _), _ in self.Automaton.SpecificationAutomaton.transition.items():
            spec_states.add(state)
        # Also add states from transition values
        for _, next_state in self.Automaton.SpecificationAutomaton.transition.items():
            spec_states.add(next_state)
        spec_states.add(self.Automaton.SpecificationAutomaton.initial_state)
        spec_states.update(self.Automaton.SpecificationAutomaton.final_states)
        
        # Sort to ensure consistent ordering (alphabetical for strings)
        self.spec_state_list = sorted(list(spec_states))
        self.spec_state_to_idx = {state: idx for idx, state in enumerate(self.spec_state_list)}
        self.idx_to_spec_state = {idx: state for idx, state in enumerate(self.spec_state_list)}


    def Start(self, is_reachability=True, max_iter=10000):
        """
        Start the controller synthesis process.
        
        Synthesizes a controller by solving either a reachability or safety problem
        using fixed-point iteration on the product automaton.
        
        Args:
            is_reachability: If True, solve reachability (reach target). 
                           If False, solve safety (stay safe forever).
            max_iter: Maximum number of iterations for fixed-point computation
            
        Returns:
            Tuple of (value_function V, controller h)
            - V[state]: For reachability: steps to reach target (-1 if unreachable)
                       For safety: 1 if safe, 0 if unsafe
            - h[state]: Control input index to apply at this state (-1 if none)
        """
        # first check if we have pre-computed results to load else compute then save
        try:
            print("Attempting to load existing controller results from file...")
            self.Load()
            print("✓ Loaded existing controller results from file.")
        
        except FileNotFoundError:
            print("✗ No existing controller results found. Computing new controller...")
            if is_reachability:
                self.V, self.h = self.SynthesisReachabilityController(max_iter)
            else:
                self.V, self.h = self.SynthesisSafetyController(max_iter)

            self.Save()
            print("✓ Computed and saved new controller results to file.")
        return self.V, self.h
        
    
    def SynthesisReachabilityController(self, max_iter):
        """
        Synthesize the reachability controller using backwards fixed-point iteration.
        
        **OPTIMIZATION v3 (Vectorized)**: Uses NumPy vectorization with inverse transitions
        for massive speedup. Processes multiple states and transitions simultaneously
        instead of looping one-by-one.
        
        **Problem**: Given initial states and target states in the product automaton,
        find the largest set R of states from which the target can be reached in finite steps,
        and for each state in R, find a control that makes progress toward the target.
        
        **Algorithm** (Backwards Iteration with Vectorization):
        1. Initialize R = target states (as NumPy boolean array)
        2. Build vectorized predecessor graph
        3. For each iteration, use vectorized operations to find newly reachable states
        4. Update R, V, h using NumPy operations (much faster than loops)
        5. Repeat until fixpoint
        
        **Value Function V**:
        - V[state] = minimum steps to reach target (0 if already in target, -1 if unreachable)
        
        **Controller h**:
        - h[state] = control index that reaches target in minimum steps (-1 if unreachable)
        
        Args:
            max_iter: Maximum number of iterations (safety against infinite loops)
            
        Returns:
            Tuple of (V, h) where:
            - V: Value function array
            - h: Controller policy array (state → control index)
        """
        start_time = time.time()
        
        # Initialize 2D arrays [spec_state_idx, sys_state_idx] for fast access
        n_spec = len(self.spec_state_list)
        n_sys = self.Automaton.total_sys_states
        
        V = np.full((n_spec, n_sys), -1, dtype=np.int32)  # -1 means unreachable
        h = np.full((n_spec, n_sys), -1, dtype=np.int32)  # -1 means no valid control
        
        # Get target states (final states in the specification automaton)
        target_spec_states = set(self.Automaton.SpecificationAutomaton.final_states)
        
        # Initialize target states with V=0 (already at target)
        num_initial_targets = 0
        for spec_state_name in target_spec_states:
            spec_state_idx = self.spec_state_to_idx[spec_state_name]
            V[spec_state_idx, :] = 0  # All sys states with target spec state
            num_initial_targets += n_sys
        
        print(f"  Initial target states: {num_initial_targets}")
        
        # Forward iteration (like reference): Check each state's successors
        for iteration in range(max_iter):
            V_old = V.copy()
            
            # Iterate over all system states
            for sys_state in range(n_sys):
                # Get labeling for this system state
                labels = self.Automaton.Labeling[sys_state]
                
                # Iterate over all spec states
                for spec_state_name in self.spec_state_list:
                    spec_state_idx = self.spec_state_to_idx[spec_state_name]
                    
                    # Skip if already reachable
                    if V[spec_state_idx, sys_state] != -1:
                        continue
                    
                    # Get next spec state based on labeling
                    next_spec_states = []
                    for label in labels:
                        if (spec_state_name, label) in self.Automaton.SpecificationAutomaton.transition:
                            next_spec = self.Automaton.SpecificationAutomaton.transition[(spec_state_name, label)]
                            next_spec_states.append(next_spec)
                    
                    if not next_spec_states:
                        continue
                    
                    # Try all controls to find one that reaches target
                    best_steps = np.inf
                    best_control = -1
                    
                    for control_idx in range(self.Automaton.SymbolicAbstraction.transition.shape[1]):
                        # Get successor range [min, max]
                        succ_range = self.Automaton.SymbolicAbstraction.transition[sys_state, control_idx, :]
                        min_succ, max_succ = succ_range[0], succ_range[1]
                        
                        if min_succ == 0 and max_succ == 0:  # No successors
                            continue
                        
                        # Check if all successors in next spec states are reachable
                        for next_spec_name in next_spec_states:
                            next_spec_idx = self.spec_state_to_idx[next_spec_name]
                            
                            # Get values for all successors in range
                            succ_values = V_old[next_spec_idx, min_succ:max_succ+1]
                            
                            # Check if all are reachable (not -1)
                            if np.all(succ_values != -1):
                                # Found a control that reaches target
                                max_steps = np.max(succ_values) + 1
                                if max_steps < best_steps:
                                    best_steps = max_steps
                                    best_control = control_idx
                    
                    if best_control != -1:
                        V[spec_state_idx, sys_state] = best_steps
                        h[spec_state_idx, sys_state] = best_control
            
            # Progress logging
            num_reachable = int(np.sum(V != -1))
            if iteration % max(1, max_iter // 10) == 0 or iteration < 5:
                elapsed = time.time() - start_time
                print(f"    Iteration {iteration}: {num_reachable} reachable states, {elapsed:.2f}s")
            
            # Check for convergence
            if np.array_equal(V, V_old):
                elapsed = time.time() - start_time
                print(f"  Reachability: Converged at iteration {iteration} ({elapsed:.2f}s)")
                break
            
            if iteration == max_iter - 1:
                elapsed = time.time() - start_time
                print(f"  Reachability: Reached max iterations ({max_iter}) ({elapsed:.2f}s)")
        
        elapsed = time.time() - start_time
        num_reachable = int(np.sum(V != -1))
        total_states = n_spec * n_sys
        print(f"  Reachability: {num_reachable} of {total_states} states reachable ({elapsed:.2f}s)")
        
        # Flatten back to 1D for compatibility with existing code
        V_flat = V.flatten()
        h_flat = h.flatten()
        return V_flat, h_flat

    def SynthesisSafetyController(self, max_iter):
        """
        Synthesize the safety controller using backwards fixed-point iteration.
        
        **OPTIMIZATION v3 (Vectorized)**: Uses NumPy vectorization to process multiple
        states and controls simultaneously.
        
        **Problem**: Given safe states and unsafe states in the product automaton,
        find the largest set S of states from which we can stay safe forever,
        and for each state in S, find a control that preserves safety (stays in S).
        
        **Algorithm** (Backwards Iteration with Vectorization):
        1. Initialize S = all safe states (as NumPy boolean array)
        2. For each state in S, check if there exists a control that keeps us in S
        3. If no such control exists, mark for removal using vectorized operations
        4. Remove unsafe states using NumPy boolean indexing
        5. Repeat until fixpoint
        
        **Value Function V**:
        - V[state] = 1 if state is safely controllable, 0 if not
        
        **Controller h**:
        - h[state] = control index that preserves safety (prefers stability)
                    -1 if no safe control exists
        
        Args:
            max_iter: Maximum number of iterations (safety against infinite loops)
            
        Returns:
            Tuple of (V, h) where:
            - V: Safety value function (1=safe, 0=unsafe)
            - h: Controller policy (state → control index for stability)
        """
        start_time = time.time()
        
        # Initialize
        V = np.zeros(self.Automaton.total_states, dtype=np.int32)  # 0 means unsafe
        h = -np.ones(self.Automaton.total_states, dtype=np.int32)  # -1 means no safe control
        
        # OPTIMIZATION v3: Use NumPy boolean array for safe_states
        safe_states = np.ones(self.Automaton.total_states, dtype=bool)
        
        # Mark target states as safe (they are our goal)
        target_states = set(self.Automaton.SpecificationAutomaton.final_states)
        target_indices = []
        for state_idx in range(self.Automaton.total_states):
            spec_state, sys_state = self._decompose_product_state(state_idx)
            # spec_state is now a string name, compare directly
            if spec_state in target_states:
                target_indices.append(state_idx)
        target_indices = np.array(target_indices, dtype=np.int32)
        V[target_indices] = 1
        
        num_safe = int(np.sum(safe_states))
        print(f"  Initial safe states: {num_safe}")
        
        # Fixed-point iteration: shrink safe set
        for iteration in range(max_iter):
            safe_old = safe_states.copy()
            unsafe_mask = np.zeros(self.Automaton.total_states, dtype=bool)
            
            # Get indices of safe states (OPTIMIZATION: vectorized indexing)
            safe_state_indices = np.where(safe_states)[0]
            
            # Process each safe state (OPTIMIZATION: batched processing)
            for state_idx in safe_state_indices:
                spec_state, sys_state = self._decompose_product_state(state_idx)
                
                # Try to find at least one safe control
                found_safe_control = False
                best_control = -1
                control_stability = np.inf
                
                # Try all possible controls
                for control_idx in range(self.Automaton.SymbolicAbstraction.transition.shape[1]):
                    # Get successors for this (state, control) pair
                    successors = self.Automaton.get_transition(
                        (spec_state, sys_state), spec_state, control_idx
                    )
                    
                    if not successors:  # No successors for this control
                        continue
                    
                    # Convert successors to state indices
                    successor_states = np.array([
                        self._compose_product_state(next_spec_state, next_sys_state)
                        for next_spec_state, next_sys_state, _ in successors
                    ], dtype=np.int32)
                    
                    # Check if ALL successors are safe (vectorized check)
                    all_successors_safe = np.all(safe_states[successor_states])
                    
                    if all_successors_safe:
                        found_safe_control = True
                        
                        # Prefer controls with fewer successors (more constrained)
                        successor_count = len(successor_states)
                        if successor_count < control_stability:
                            control_stability = successor_count
                            best_control = control_idx
                
                # If no safe control found, mark as unsafe
                if not found_safe_control:
                    unsafe_mask[state_idx] = True
                    V[state_idx] = 0
                else:
                    V[state_idx] = 1
                    h[state_idx] = best_control
            
            # Remove unsafe states using vectorized operation
            safe_states = safe_states & ~unsafe_mask
            
            # Progress logging with vectorized operations
            num_safe = int(np.sum(safe_states))
            num_removed = int(np.sum(unsafe_mask))
            if iteration % max(1, max_iter // 10) == 0 or iteration < 5:
                elapsed = time.time() - start_time
                print(f"    Iteration {iteration}: {num_safe} safe states, removed {num_removed}, {elapsed:.2f}s")
            
            # Early termination if no changes (vectorized check)
            if np.array_equal(safe_states, safe_old):
                elapsed = time.time() - start_time
                print(f"  Safety: Converged at iteration {iteration} ({elapsed:.2f}s)")
                break
            
            if iteration == max_iter - 1:
                elapsed = time.time() - start_time
                print(f"  Safety: Reached max iterations ({max_iter}) ({elapsed:.2f}s)")
        
        elapsed = time.time() - start_time
        num_safe = int(np.sum(safe_states))
        print(f"  Safety: {num_safe} of {self.Automaton.total_states} states safely controllable ({elapsed:.2f}s)")
        return V, h

    def _decompose_product_state(self, product_state_idx):
        """
        Decompose a product automaton state index into (spec_state, sys_state).
        
        Product states are organized as: spec_state_idx * N_sys_states + sys_state
        
        Args:
            product_state_idx: Index in product automaton
            
        Returns:
            Tuple of (spec_state_name, sys_state_idx)
            - spec_state_name: String name like 'a', 'b', 'c', 'd', 'e'
            - sys_state_idx: Integer index in system abstraction
        """
        N_sys = self.Automaton.total_sys_states
        spec_state_idx = product_state_idx // N_sys
        sys_state = product_state_idx % N_sys
        spec_state_name = self.idx_to_spec_state[spec_state_idx]
        return spec_state_name, sys_state

    def _compose_product_state(self, spec_state, sys_state):
        """
        Compose a product state index from (spec_state, sys_state).
        
        Product states are organized as: spec_state_idx * N_sys_states + sys_state
        
        Args:
            spec_state: Specification automaton state (can be name string or index)
            sys_state: System abstraction state index
            
        Returns:
            Product state index
        """
        # Handle both string names and integer indices
        if isinstance(spec_state, str):
            spec_state_idx = self.spec_state_to_idx[spec_state]
        else:
            spec_state_idx = spec_state
        
        N_sys = self.Automaton.total_sys_states
        return spec_state_idx * N_sys + sys_state

    def _build_inverse_transition_map(self):
        """
        Build inverse transition map: for each state, list all (predecessor, control) pairs.
        
        **OPTIMIZATION v2**: Pre-compute which states can reach each target in one step.
        This enables efficient backward reachability without checking all states every iteration.
        
        Returns:
            Dictionary: {target_state: [(predecessor_state, control_idx), ...]}
        """
        inverse_map = {}
        
        # Iterate through all states and controls to build inverse map
        for state_idx in range(self.Automaton.total_states):
            spec_state, sys_state = self._decompose_product_state(state_idx)
            
            for control_idx in range(self.Automaton.SymbolicAbstraction.transition.shape[1]):
                successors = self.Automaton.get_transition(
                    (spec_state, sys_state), spec_state, control_idx
                )
                
                for next_spec_state, next_sys_state, _ in successors:
                    next_product_state = self._compose_product_state(next_spec_state, next_sys_state)
                    
                    if next_product_state not in inverse_map:
                        inverse_map[next_product_state] = []
                    
                    inverse_map[next_product_state].append((state_idx, control_idx))
        
        return inverse_map

    # Saving and loading for time efficiency (avoiding recomputation)
    def Load(self):
        """
        Load pre-computed value function and controller from file.
        
        Returns:
            self (for method chaining)
        """
        self.V = np.loadtxt(f'{self.model_dir}/V_result.csv', delimiter=',')
        self.h = np.loadtxt(f'{self.model_dir}/h_result.csv', delimiter=',')
        print("Loaded saved results successfully.")

    def Save(self):
        """
        Save the computed value function and controller to file.
        """
        np.savetxt(f'{self.model_dir}/V_result.csv', self.V, delimiter=',')
        np.savetxt(f'{self.model_dir}/h_result.csv', self.h, delimiter=',')

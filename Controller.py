import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

class ControllerSynthesis:

    def __init__(self, Automaton, model_dir='./Models', num_threads=None):
        """
        Initialize the controller synthesis engine.
        
        Args:
            Automaton: Instance of the product automaton (system × specification)
            model_dir: Directory to save/load model files (default: './Models')
            num_threads: Number of CPU threads to use for parallel synthesis (default: None = auto-detect)
        """
        self.Automaton = Automaton
        self.model_dir = model_dir
        self.V = None
        self.h = None
        self.num_threads = num_threads or os.cpu_count() or 4  # Default to CPU count or 4


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
        
        **OPTIMIZATION v4 (Multi-threaded Vectorized)**: Uses ThreadPoolExecutor for parallel
        processing of state transitions with NumPy vectorization.
        
        **Problem**: Given initial states and target states in the product automaton,
        find the largest set R of states from which the target can be reached in finite steps,
        and for each state in R, find a control that makes progress toward the target.
        
        **Algorithm** (Backwards Iteration with Multi-threading):
        1. Initialize R = target states (as NumPy boolean array)
        2. Build vectorized predecessor graph
        3. For each iteration, use ThreadPoolExecutor to process states in parallel
        4. Each thread processes a batch of states using vectorized operations
        5. Combine results with thread-safe operations
        6. Repeat until fixpoint
        
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
        
        # Initialize
        V = -np.ones(self.Automaton.total_states, dtype=np.int32)  # -1 means unreachable
        h = -np.ones(self.Automaton.total_states, dtype=np.int32)  # -1 means no valid control
        
        # Get target states (final states in the specification automaton)
        target_spec_states = set(self.Automaton.SpecificationAutomaton.final_states)
        
        # OPTIMIZATION v4: Use NumPy boolean array for R instead of set
        # This enables vectorized operations
        R = np.zeros(self.Automaton.total_states, dtype=bool)
        for state_idx in range(self.Automaton.total_states):
            spec_state, sys_state = self._decompose_product_state(state_idx)
            if spec_state in target_spec_states:
                R[state_idx] = True
                V[state_idx] = 0  # Already at target
        
        num_initial_targets = np.sum(R)
        print(f"  Initial target states: {int(num_initial_targets)}")
        print(f"  Using {self.num_threads} CPU threads for parallel synthesis...")
        
        # Use pre-computed inverse transition map from AbstractSpace
        print(f"  Using pre-computed inverse transition map...")
        inverse_transition = self.Automaton.SymbolicAbstraction.inverse_transition
        
        # Thread-safe lock for atomic state updates (single lock prevents deadlocks)
        state_lock = Lock()
        
        def process_state_batch(states_batch):
            """Process a batch of states in parallel and return updates"""
            local_updates = []
            
            for target_state in states_batch:
                if target_state in inverse_transition:
                    predecessors = inverse_transition[target_state]
                    
                    for predecessor_state, control_idx in predecessors:
                        if not R[predecessor_state]:  # Only process if not already reachable
                            # Check if this control actually reaches target
                            spec_state, sys_state = self._decompose_product_state(predecessor_state)
                            successors = self.Automaton.get_transition(
                                (spec_state, sys_state), spec_state, control_idx
                            )
                            
                            # Check if any successor is in R (vectorized check)
                            if successors:
                                successor_states = np.array([
                                    self._compose_product_state(next_spec_state, next_sys_state)
                                    for next_spec_state, next_sys_state, _ in successors
                                ])
                                
                                if np.any(R[successor_states]):
                                    # Find best successor (minimum steps)
                                    reachable_successors = successor_states[R[successor_states]]
                                    steps_to_target = np.min(V[reachable_successors]) + 1
                                    
                                    # Only update if this is better than current
                                    if V[predecessor_state] == -1 or steps_to_target < V[predecessor_state]:
                                        local_updates.append((predecessor_state, steps_to_target, control_idx))
            
            return local_updates
        
        # OPTIMIZATION v4: Vectorized fixed-point iteration with multi-threading
        iteration = 0
        while iteration < max_iter:
            R_old = R.copy()
            newly_reachable = np.zeros(self.Automaton.total_states, dtype=bool)
            
            # Find states that changed this iteration to minimize work
            # Only check predecessors of newly reachable states
            states_to_check_indices = np.where(R & ~R_old)[0] if iteration > 0 else np.where(R)[0]
            
            if len(states_to_check_indices) > 0:
                # Split work into batches for multi-threading
                batch_size = max(1, len(states_to_check_indices) // (self.num_threads * 4))
                batches = [states_to_check_indices[i:i+batch_size] 
                          for i in range(0, len(states_to_check_indices), batch_size)]
                
                # Process batches in parallel
                all_updates = []
                with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                    futures = [executor.submit(process_state_batch, batch) for batch in batches]
                    for future in as_completed(futures):
                        all_updates.extend(future.result())
                
                # Apply all updates with thread-safe atomic lock
                for predecessor_state, steps_to_target, control_idx in all_updates:
                    with state_lock:  # Single lock ensures V and h update atomically
                        if V[predecessor_state] == -1 or steps_to_target < V[predecessor_state]:
                            V[predecessor_state] = steps_to_target
                            h[predecessor_state] = control_idx
                            newly_reachable[predecessor_state] = True
            
            R = R | newly_reachable  # Vectorized union (OPTIMIZATION)
            
            # Progress logging with vectorized operations
            num_reachable = int(np.sum(R))
            if iteration % max(1, max_iter // 10) == 0 or iteration < 5:
                elapsed = time.time() - start_time
                print(f"    Iteration {iteration}: {num_reachable} reachable states, {elapsed:.2f}s")
            
            # Check for convergence (vectorized comparison)
            if np.array_equal(R, R_old):
                elapsed = time.time() - start_time
                print(f"  Reachability: Converged at iteration {iteration} ({elapsed:.2f}s)")
                break
            
            iteration += 1
            
            if iteration >= max_iter:
                elapsed = time.time() - start_time
                print(f"  Reachability: Reached max iterations ({max_iter}) ({elapsed:.2f}s)")
                break
        
        elapsed = time.time() - start_time
        num_reachable = int(np.sum(R))
        print(f"  Reachability: {num_reachable} of {self.Automaton.total_states} states reachable ({elapsed:.2f}s)")
        return V, h

    def SynthesisSafetyController(self, max_iter):
        """
        Synthesize the safety controller using backwards fixed-point iteration.
        
        **OPTIMIZATION v4 (Multi-threaded Vectorized)**: Uses ThreadPoolExecutor for parallel
        processing of state transitions with NumPy vectorization.
        
        **Problem**: Given safe states and unsafe states in the product automaton,
        find the largest set S of states from which we can stay safe forever,
        and for each state in S, find a control that preserves safety (stays in S).
        
        **Algorithm** (Backwards Iteration with Multi-threading):
        1. Initialize S = all safe states (as NumPy boolean array)
        2. For each state in S, check if there exists a control that keeps us in S
        3. Use ThreadPoolExecutor to process multiple states in parallel
        4. If no such control exists, mark for removal using vectorized operations
        5. Remove unsafe states using NumPy boolean indexing
        6. Repeat until fixpoint
        
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
        
        # OPTIMIZATION v4: Use NumPy boolean array for safe_states
        safe_states = np.ones(self.Automaton.total_states, dtype=bool)
        
        # Mark target states as safe (they are our goal)
        target_states = self.Automaton.SpecificationAutomaton.final_states
        target_indices = np.array([
            state_idx for state_idx in range(self.Automaton.total_states)
            if (state_idx // self.Automaton.total_sys_states) in target_states
        ], dtype=np.int32)
        V[target_indices] = 1
        
        num_safe = int(np.sum(safe_states))
        print(f"  Initial safe states: {num_safe}")
        print(f"  Using {self.num_threads} CPU threads for parallel synthesis...")
        
        # Thread-safe lock for atomic state updates
        state_lock = Lock()
        
        def process_safe_states_batch(state_indices_batch):
            """Process a batch of safe states in parallel"""
            local_updates = []
            
            for state_idx in state_indices_batch:
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
                    local_updates.append((state_idx, False, -1))
                else:
                    local_updates.append((state_idx, True, best_control))
            
            return local_updates
        
        # Fixed-point iteration: shrink safe set
        iteration = 0
        while iteration < max_iter:
            safe_old = safe_states.copy()
            
            # Get indices of safe states (OPTIMIZATION: vectorized indexing)
            safe_state_indices = np.where(safe_states)[0]
            
            if len(safe_state_indices) > 0:
                # Split work into batches for multi-threading
                batch_size = max(1, len(safe_state_indices) // (self.num_threads * 4))
                batches = [safe_state_indices[i:i+batch_size] 
                          for i in range(0, len(safe_state_indices), batch_size)]
                
                # Process batches in parallel
                all_updates = []
                with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                    futures = [executor.submit(process_safe_states_batch, batch) for batch in batches]
                    for future in as_completed(futures):
                        all_updates.extend(future.result())
                
                # Apply updates
                unsafe_mask = np.zeros(self.Automaton.total_states, dtype=bool)
                for state_idx, is_safe, best_control in all_updates:
                    if not is_safe:
                        unsafe_mask[state_idx] = True
                        V[state_idx] = 0
                    else:
                        V[state_idx] = 1
                        h[state_idx] = best_control
                
                # Remove unsafe states using vectorized operation
                safe_states = safe_states & ~unsafe_mask
            
            # Progress logging with vectorized operations
            num_safe = int(np.sum(safe_states))
            num_removed = int(np.sum(~safe_old & safe_states == False))
            log_interval = 10 if max_iter == float('inf') else max(1, int(max_iter // 10))
            if iteration % log_interval == 0 or iteration < 5:
                elapsed = time.time() - start_time
                print(f"    Iteration {iteration}: {num_safe} safe states, removed {num_removed}, {elapsed:.2f}s")
            
            # Early termination if no changes (vectorized check)
            if np.array_equal(safe_states, safe_old):
                elapsed = time.time() - start_time
                print(f"  Safety: Converged at iteration {iteration} ({elapsed:.2f}s)")
                break
            
            iteration += 1
            
            if iteration >= max_iter:
                elapsed = time.time() - start_time
                print(f"  Safety: Reached max iterations ({max_iter}) ({elapsed:.2f}s)")
                break
        
        elapsed = time.time() - start_time
        num_safe = int(np.sum(safe_states))
        print(f"  Safety: {num_safe} of {self.Automaton.total_states} states safely controllable ({elapsed:.2f}s)")
        return V, h

    def _decompose_product_state(self, product_state_idx):
        """
        Decompose a product automaton state index into (spec_state, sys_state).
        
        Product states are organized as: spec_state * N_sys_states + sys_state
        
        Args:
            product_state_idx: Index in product automaton
            
        Returns:
            Tuple of (spec_state_idx, sys_state_idx)
        """
        N_sys = self.Automaton.total_sys_states
        spec_state = product_state_idx // N_sys
        sys_state = product_state_idx % N_sys
        return spec_state, sys_state

    def _compose_product_state(self, spec_state, sys_state):
        """
        Compose a product state index from (spec_state, sys_state).
        
        Product states are organized as: spec_state * N_sys_states + sys_state
        
        Args:
            spec_state: Specification automaton state index
            sys_state: System abstraction state index
            
        Returns:
            Product state index
        """
        N_sys = self.Automaton.total_sys_states
        return spec_state * N_sys + sys_state

    def _build_inverse_transition_map(self):
        """
        DEPRECATED: Inverse transition map is now pre-computed in AbstractSpace.
        
        This method is kept for backward compatibility but should not be used.
        Use: self.Automaton.SymbolicAbstraction.inverse_transition instead
        """
        raise NotImplementedError(
            "Inverse transition map is now pre-computed in AbstractSpace. "
            "Use: self.Automaton.SymbolicAbstraction.inverse_transition"
        )

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
        Automatically creates the model directory if it doesn't exist.
        """
        # Ensure directory exists before saving
        os.makedirs(self.model_dir, exist_ok=True)
        
        np.savetxt(f'{self.model_dir}/V_result.csv', self.V, delimiter=',')
        np.savetxt(f'{self.model_dir}/h_result.csv', self.h, delimiter=',')

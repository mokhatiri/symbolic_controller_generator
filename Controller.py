import numpy as np

class ControllerSynthesis:

    def __init__(self, Automaton):
        """
        Initialize the controller synthesis engine.
        
        Args:
            Automaton: Instance of the product automaton (system × specification)
        """
        self.Automaton = Automaton
        self.V = None
        self.h = None


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
        
        **Problem**: Given initial states and target states in the product automaton,
        find the largest set R of states from which the target can be reached in finite steps,
        and for each state in R, find a control that makes progress toward the target.
        
        **Algorithm** (Backwards Iteration):
        1. Initialize R = target states
        2. For each state NOT in R, check if it can reach R in one step with some control
        3. If yes, add state to R and record the fastest control to the target
        4. Repeat until fixpoint (no new states added)
        
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
        # Initialize
        V = -np.ones(self.Automaton.total_states, dtype=int)  # -1 means unreachable
        h = -np.ones(self.Automaton.total_states, dtype=int)  # -1 means no valid control
        
        # Get target states (final states in the specification automaton)
        target_spec_states = set(self.Automaton.SpecificationAutomaton.final_states)
        
        # Build set of product states that are in target spec states
        R = set()
        for state_idx in range(self.Automaton.total_states):
            spec_state, sys_state = self._decompose_product_state(state_idx)
            if spec_state in target_spec_states:
                R.add(state_idx)
                V[state_idx] = 0  # Already at target
        
        # Fixed-point iteration: expand reachable set backwards
        for iteration in range(max_iter):
            R_old = R.copy()
            
            # Check all states not yet in R
            for state_idx in range(self.Automaton.total_states):
                if state_idx not in R:
                    spec_state, sys_state = self._decompose_product_state(state_idx)
                    
                    # Try all possible controls from this state
                    min_steps = float('inf')
                    best_control = -1
                    
                    # Get all possible transition from this state
                    for control_idx in range(self.Automaton.SymbolicAbstraction.transition.shape[1]):
                        # Get successors for this (state, control) pair
                        successors = self.Automaton.get_transition(
                            (spec_state, sys_state), spec_state, control_idx
                        )
                        
                        # Check if any successor is in R
                        for next_spec_state, next_sys_state, _ in successors:
                            next_product_state = self._compose_product_state(next_spec_state, next_sys_state)
                            
                            if next_product_state in R:
                                # This control reaches target!
                                steps_to_target = V[next_product_state] + 1
                                
                                # Prefer control with minimum steps (fastest path)
                                if steps_to_target < min_steps:
                                    min_steps = steps_to_target
                                    best_control = control_idx
                    
                    # If found a control that reaches target, add to R
                    if best_control != -1:
                        R.add(state_idx)
                        V[state_idx] = min_steps
                        h[state_idx] = best_control
            
            # Check for convergence
            if R == R_old:
                print(f"Reachability: Converged at iteration {iteration}")
                break
            
            if iteration == max_iter - 1:
                print(f"Reachability: Reached max iterations ({max_iter})")
        
        print(f"Reachability: {len(R)} of {self.Automaton.total_states} states are reachable")
        return V, h

    def SynthesisSafetyController(self, max_iter):
        """
        Synthesize the safety controller using backwards fixed-point iteration.
        
        **Problem**: Given safe states and unsafe states in the product automaton,
        find the largest set S of states from which we can stay safe forever,
        and for each state in S, find a control that preserves safety (stays in S).
        
        **Algorithm** (Backwards Iteration):
        1. Initialize S = all safe states
        2. For each state in S, check if there exists a control that keeps us in S
        3. If no such control exists, remove state from S
        4. Repeat until fixpoint (no states removed)
        
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
        # Initialize
        V = np.zeros(self.Automaton.total_states, dtype=int)  # 0 means unsafe
        h = -np.ones(self.Automaton.total_states, dtype=int)  # -1 means no safe control
        
        # Define safe states (all states that are not in unsafe regions)
        # For now, assume all states in valid spec states are initially safe
        safe_states = set(range(self.Automaton.total_states))
        
        # Mark target states as safe (they are our goal)
        target_states = set(self.Automaton.SpecificationAutomaton.final_states)
        for state in target_states:
            V[state] = 1
        
        # Fixed-point iteration: shrink safe set
        for iteration in range(max_iter):
            safe_old = safe_states.copy()
            states_to_remove = set()
            
            # Check each safe state
            for state_idx in list(safe_states):
                spec_state, sys_state = self._decompose_product_state(state_idx)
                
                # Try to find at least one safe control
                found_safe_control = False
                best_control = -1
                control_stability = float('inf')
                
                # Try all possible controls
                for control_idx in range(self.Automaton.SymbolicAbstraction.transition.shape[1]):
                    # Get successors for this (state, control) pair
                    successors = self.Automaton.get_transition(
                        (spec_state, sys_state), spec_state, control_idx
                    )
                    
                    # Check if ALL successors are safe
                    all_successors_safe = True
                    successor_count = 0
                    
                    for next_spec_state, next_sys_state, _ in successors:
                        successor_count += 1
                        next_product_state = self._compose_product_state(next_spec_state, next_sys_state)
                        
                        if next_product_state not in safe_states:
                            all_successors_safe = False
                            break
                    
                    if all_successors_safe and successor_count > 0:
                        found_safe_control = True
                        
                        # Prefer controls with fewer successors (more constrained, more stable)
                        # This biases toward "safer" controls that have less nondeterminism
                        if successor_count < control_stability:
                            control_stability = successor_count
                            best_control = control_idx
                
                # If no safe control found, remove from safe set
                if not found_safe_control:
                    states_to_remove.add(state_idx)
                    V[state_idx] = 0
                else:
                    V[state_idx] = 1
                    h[state_idx] = best_control
            
            # Remove unsafe states
            safe_states -= states_to_remove
            
            # Check for convergence
            if safe_states == safe_old:
                print(f"Safety: Converged at iteration {iteration}")
                break
            
            if iteration == max_iter - 1:
                print(f"Safety: Reached max iterations ({max_iter})")
        
        print(f"Safety: {len(safe_states)} of {self.Automaton.total_states} states are safely controllable")
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

    # Saving and loading for time efficiency (avoiding recomputation)
    def Load(self):
        """
        Load pre-computed value function and controller from file.
        
        Returns:
            self (for method chaining)
        """
        self.V = np.loadtxt('./Models/V_result.csv', delimiter=',')
        self.h = np.loadtxt('./Models/h_result.csv', delimiter=',')
        print("Loaded saved results successfully.")

    def Save(self):
        """
        Save the computed value function and controller to file.
        """
        np.savetxt('./Models/V_result.csv', self.V, delimiter=',')
        np.savetxt('./Models/h_result.csv', self.h, delimiter=',')

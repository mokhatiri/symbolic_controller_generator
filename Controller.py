import numpy as np
import time
import os


class ControllerSynthesis:
    """
    Controller synthesis using ApplySecurity then ApplyReachability fixed-point algorithms.
    
    Memory-efficient: stores (min, max) ranges instead of expanded lists.
    Controller h: dict[(sys_state, spec_state)] = control_idx
    
    Works with N-dimensional state spaces.
    """

    def __init__(self, ProdAutomaton, model_dir='./Models'):
        """
        Initialize the controller synthesis engine.
        """
        self.Automaton = ProdAutomaton
        self.model_dir = model_dir
        
        # Controller dictionary: h[(sys_state, spec_state)] = control_idx
        self.h = {}
        
        # V and h for numpy-based storage (for compatibility)
        self.V = None
        self.h_array = None
        
        # Cache discretisation reference
        self.disc = self.Automaton.SymbolicAbstraction.Discretisation
        self.N_sys = self.Automaton.total_sys_states
        self.N_u = self.Automaton.total_controls
        self.N_spec = self.Automaton.total_spec_states


    def _get_successors_range(self, sys_state, control_idx):
        """
        Get the range of successor states for a given system state and control.
        
        Returns:
            Tuple (min_succ, max_succ) or None if transition is invalid
        """
        transition = self.Automaton.SymbolicAbstraction.transition
        min_succ = int(transition[sys_state, control_idx, 0])
        max_succ = int(transition[sys_state, control_idx, 1])
        
        # A valid transition has min_succ <= max_succ
        # Invalid transitions are indicated by min > max (e.g., both 0 could be valid for state 0)
        if min_succ <= max_succ:
            return (min_succ, max_succ)
        return None


    def _get_trash_spec_states(self):
        """
        Get the set of trash/unsafe specification states.
        
        Trash states are non-accepting states that have no outgoing transitions
        to accepting states, or states explicitly marked as trash.
        
        Returns:
            Set of trash specification state names
        """
        spec_automaton = self.Automaton.SpecificationAutomaton
        final_states = set(spec_automaton.final_states)
        all_states = set(spec_automaton.state_list)
        
        # Find states that can never reach an accepting state
        # Simple heuristic: states that only transition to themselves
        trash_states = set()
        
        for state in all_states:
            if state in final_states:
                continue  # Accepting states are not trash
            
            # Check if this state only transitions to itself
            outgoing = spec_automaton.get_outgoing_transitions(state)
            if all(next_state == state for (_, next_state) in outgoing):
                trash_states.add(state)
        
        return trash_states


    def _get_target_states(self, product_states):
        """
        Get target states (product states with accepting specification).
        
        In the product automaton, target states are those where the specification
        component is in a final/accepting state.
        
        Args:
            product_states: Set of reachable product states (sys_state, spec_state)
            
        Returns:
            Set of target product states
        """
        spec_automaton = self.Automaton.SpecificationAutomaton
        final_states = set(spec_automaton.final_states)
        
        # Target states are product states with accepting specification state
        target_states = set()
        for (sys_state, spec_state) in product_states:
            if spec_state in final_states:
                target_states.add((sys_state, spec_state))
        
        return target_states


    def _all_successors_in_set(self, sys_state, control_idx, safe_set, spec_state):
        """
        Check if ALL successors of (sys_state, spec_state) under control_idx are in safe_set.
        
        This handles the product automaton transitions.
        
        Args:
            sys_state: Current system state index
            control_idx: Control input index
            safe_set: Set of safe product states (sys_state, spec_state)
            spec_state: Current specification state name
            
        Returns:
            True if all successors are in safe_set, False otherwise
        """
        succ_range = self._get_successors_range(sys_state, control_idx)
        if succ_range is None:
            return False
        
        min_succ, max_succ = succ_range
        spec_automaton = self.Automaton.SpecificationAutomaton
        
        for next_sys in range(min_succ, max_succ + 1):
            # Get labels for next system state
            labels = self.Automaton.Labeling[next_sys]
            
            # For each label, check the specification transition
            found_valid = False
            for label in labels:
                next_spec = spec_automaton.get_next_state(spec_state, label)
                if next_spec is not None:
                    # Check if successor is in safe set
                    if (next_sys, next_spec) in safe_set:
                        found_valid = True
                        break
            
            if not found_valid:
                return False
        
        return True


    def _all_sys_successors_in_domain(self, sys_state, control_idx, domain):
        """
        Check if ALL system successors under control_idx stay within domain.
        
        This is used for the initial security check before product automaton.
        
        Args:
            sys_state: Current system state index
            control_idx: Control input index
            domain: Set of valid system states
            
        Returns:
            True if all successors are in domain, False otherwise
        """
        succ_range = self._get_successors_range(sys_state, control_idx)
        if succ_range is None:
            return False
        
        min_succ, max_succ = succ_range
        
        # Check all successors are within domain
        for next_sys in range(min_succ, max_succ + 1):
            if next_sys not in domain:
                return False
        
        return True


    def ApplyReachability(self, target_states, safe_domain, max_iter=10000):
        """
        Apply reachability fixed-point to find states that can reach target.
        
        Uses backward reachability from target states while staying within safe_domain.
        
        Args:
            target_states: Set of target product states (sys_state, spec_state)
            safe_domain: Set of safe product states
            max_iter: Maximum iterations
            
        Returns:
            Set of reachable product states
        """
        self.h = {}
        
        # Initialize with target states that are in safe domain
        R = target_states.intersection(safe_domain)
        
        # For target states, we assign a default control (first valid one)
        for (sys_state, spec_state) in R:
            for control_idx in range(self.N_u):
                if self._get_successors_range(sys_state, control_idx) is not None:
                    self.h[(sys_state, spec_state)] = control_idx
                    break
        
        print(f"    ApplyReachability: Initial target set size: {len(R)}")
        
        for iteration in range(max_iter):
            # Find predecessors that can reach R
            pred = self._predecessor(R, safe_domain)
            
            R_new = R.union(pred)
            
            if R_new == R:
                print(f"    ApplyReachability converged after {iteration + 1} iterations")
                break
            
            R = R_new
            
            if (iteration + 1) % 10 == 0:
                print(f"    ApplyReachability iteration {iteration + 1}: {len(R)} states")
        
        return R


    def _predecessor(self, R, safe_domain):
        """
        Find all states in safe_domain that have a control making ALL successors go to R.
        
        Args:
            R: Current reachable set
            safe_domain: Set of safe product states
            
        Returns:
            Set of predecessor states
        """
        pred = set()
        
        for (sys_state, spec_state) in safe_domain:
            if (sys_state, spec_state) in R:
                continue  # Already in R
            
            # Check if any control keeps all successors in R
            for control_idx in range(self.N_u):
                if self._all_successors_in_R(sys_state, control_idx, R, spec_state):
                    pred.add((sys_state, spec_state))
                    # Store the control if not already set
                    if (sys_state, spec_state) not in self.h:
                        self.h[(sys_state, spec_state)] = control_idx
                    break
        
        return pred


    def _all_successors_in_R(self, sys_state, control_idx, R, spec_state):
        """
        Check if ALL successors of (sys_state, spec_state) under control_idx are in R.
        
        Args:
            sys_state: Current system state index
            control_idx: Control input index
            R: Reachable set of product states
            spec_state: Current specification state name
            
        Returns:
            True if all successors are in R, False otherwise
        """
        succ_range = self._get_successors_range(sys_state, control_idx)
        if succ_range is None:
            return False
        
        min_succ, max_succ = succ_range
        spec_automaton = self.Automaton.SpecificationAutomaton
        
        for next_sys in range(min_succ, max_succ + 1):
            # Get labels for next system state
            labels = self.Automaton.Labeling[next_sys]
            
            found_in_R = False
            for label in labels:
                next_spec = spec_automaton.get_next_state(spec_state, label)
                if next_spec is not None and (next_sys, next_spec) in R:
                    found_in_R = True
                    break
            
            if not found_in_R:
                return False
        
        return True


    def _convert_h_to_arrays(self):
        """
        Convert the controller dictionary to numpy arrays for compatibility.
        
        Creates:
            V: Set of winning states as boolean array [spec_states, sys_states]
            h_array: Control array [spec_states, sys_states], -1 for no control
        """
        spec_automaton = self.Automaton.SpecificationAutomaton
        
        # Initialize arrays
        self.V = np.zeros((self.N_spec, self.N_sys), dtype=bool)
        self.h_array = np.full((self.N_spec, self.N_sys), -1, dtype=int)
        
        for (sys_state, spec_state), control_idx in self.h.items():
            spec_idx = spec_automaton.get_state_index(spec_state)
            self.V[spec_idx, sys_state] = True
            self.h_array[spec_idx, sys_state] = control_idx


    def Start(self, is_reachability=True, max_iter=10000):
        """
        Start controller synthesis following the original pattern:
        1. ApplySecurity on system states (keep states within bounds)
        2. Build reachable product automaton from initial spec state
        3. Apply safety to remove product states leading to trash
        4. ApplyReachability from accepting states
        """
        # Check for cached results
        try:
            print("Attempting to load existing controller results from file...")
            self.Load()
            print("✓ Loaded existing controller results from file.")
            return self.V, self.h_array
        except FileNotFoundError:
            pass
        
        print("✗ No existing controller results found. Computing new controller...")
        start_time = time.time()
        
        # Step 1: Build the domain (all valid system states with at least one valid transition)
        domain = set()
        for sys_state in range(self.N_sys):
            for control_idx in range(self.N_u):
                if self._get_successors_range(sys_state, control_idx) is not None:
                    domain.add(sys_state)
                    break
        print(f"  Step 1 - Initial domain: {len(domain)} system states with valid transitions")
        
        # Step 2: Apply Security on SYSTEM states - find states that stay within bounds
        safe_sys_states = self._apply_security_system(domain)
        print(f"  Step 2 - After ApplySecurity (system): {len(safe_sys_states)} safe system states")
        
        # Step 3: Build the reachable product automaton from initial spec state
        # This is like ToProdAutomate in original
        trash_states = self._get_trash_spec_states()
        print(f"  Trash specification states: {trash_states}")
        
        product_states = self._build_product_automaton(safe_sys_states, trash_states)
        print(f"  Step 3 - Product automaton: {len(product_states)} reachable product states")
        
        # Step 4: Define target states (product states with accepting spec)
        target_states = self._get_target_states(product_states)
        print(f"  Step 4 - Target states (accepting): {len(target_states)}")
        
        # Step 5: Apply Reachability from target states
        R = self.ApplyReachability(target_states, product_states, max_iter)
        print(f"  Step 5 - After ApplyReachability: {len(R)} winning states")
        
        elapsed = time.time() - start_time
        print(f"  Controller synthesis completed in {elapsed:.2f}s")
        
        # Convert to numpy arrays for compatibility
        self._convert_h_to_arrays()
        
        # Save results
        self.Save()
        print("✓ Computed and saved new controller results to file.")
        
        return self.V, self.h_array


    def _apply_security_system(self, domain):
        """
        Apply security fixed-point on system states only.
        Finds the maximal set of states that can stay within bounds.
        """
        R = domain.copy()
        
        iteration = 0
        while True:
            iteration += 1
            R_new = set()
            
            for sys_state in R:
                # Check if there exists a control keeping all successors in R
                has_safe_control = False
                for control_idx in range(self.N_u):
                    if self._all_sys_successors_in_domain(sys_state, control_idx, R):
                        has_safe_control = True
                        break
                
                if has_safe_control:
                    R_new.add(sys_state)
            
            if R_new == R:
                break
            
            R = R_new
            
            if iteration % 10 == 0:
                print(f"    ApplySecurity iteration {iteration}: {len(R)} states")
        
        print(f"    ApplySecurity converged after {iteration} iterations")
        return R


    def _build_product_automaton(self, safe_sys_states, trash_states):
        """
        Build the reachable product automaton via BFS from initial spec state.
        This mimics ToProdAutomate from the original implementation.
        
        Only includes product states (sys_state, spec_state) that are:
        1. Reachable from any system state with initial spec state
        2. Not in trash specification states
        """
        from collections import deque
        
        spec_automaton = self.Automaton.SpecificationAutomaton
        initial_spec = spec_automaton.initial_state
        
        # Build product automaton via BFS
        visited = set()
        queue = deque()
        
        # Initialize with all safe system states at initial spec state
        for sys_state in safe_sys_states:
            prod_state = (sys_state, initial_spec)
            if initial_spec not in trash_states:
                queue.append(prod_state)
                visited.add(prod_state)
        
        while queue:
            (sys_state, spec_state) = queue.popleft()
            
            # For each control, explore transitions
            for control_idx in range(self.N_u):
                succ_range = self._get_successors_range(sys_state, control_idx)
                if succ_range is None:
                    continue
                
                min_succ, max_succ = succ_range
                
                for next_sys in range(min_succ, max_succ + 1):
                    if next_sys not in safe_sys_states:
                        continue
                    
                    # Get labels and compute next spec state
                    labels = self.Automaton.Labeling[next_sys]
                    for label in labels:
                        next_spec = spec_automaton.get_next_state(spec_state, label)
                        if next_spec is not None and next_spec not in trash_states:
                            next_prod = (next_sys, next_spec)
                            if next_prod not in visited:
                                visited.add(next_prod)
                                queue.append(next_prod)
        
        return visited


    def Save(self):
        """
        Save the controller results to files.
        """
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save V (winning states)
        np.save(os.path.join(self.model_dir, "V.npy"), self.V)
        
        # Save h_array (control mapping)
        np.save(os.path.join(self.model_dir, "h_array.npy"), self.h_array)
        
        print(f"  Controller saved to {self.model_dir}")


    def Load(self):
        """
        Load the controller results from files.
        
        Raises:
            FileNotFoundError: If controller files don't exist
        """
        v_path = os.path.join(self.model_dir, "V.npy")
        h_path = os.path.join(self.model_dir, "h_array.npy")
        
        if not os.path.exists(v_path) or not os.path.exists(h_path):
            raise FileNotFoundError("Controller files not found")
        
        self.V = np.load(v_path)
        self.h_array = np.load(h_path)
        
        # Reconstruct h dictionary from h_array
        spec_automaton = self.Automaton.SpecificationAutomaton
        self.h = {}
        for spec_idx in range(self.N_spec):
            spec_state = spec_automaton.get_state_name(spec_idx)
            for sys_state in range(self.N_sys):
                control_idx = self.h_array[spec_idx, sys_state]
                if control_idx >= 0:
                    self.h[(sys_state, spec_state)] = control_idx
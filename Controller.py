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


    def _get_target_states(self, safe_domain):
        """
        Get the set of target product states (system_state, spec_state) that are in accepting states.
        
        Args:
            safe_domain: Set of safe product states (sys_state, spec_state)
            
        Returns:
            Set of (sys_state, spec_state_name) tuples that are target states
        """
        spec_automaton = self.Automaton.SpecificationAutomaton
        final_states = set(spec_automaton.final_states)
        
        # Target states are those in safe_domain that have an accepting specification state
        target_states = set()
        for (sys_state, spec_state) in safe_domain:
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


    def ApplySecurity(self, domain, trash_states):
        """
        Apply security fixed-point to find the maximal safe domain.
        
        This follows the original pattern:
        1. First find system states that can stay within bounds
        2. Then build product states excluding trash specification states
        
        Args:
            domain: Set of initial system states
            trash_states: Set of trash specification state names
            
        Returns:
            Set of safe product states (sys_state, spec_state)
        """
        # Step 1: Apply security on SYSTEM states only (like original ApplySecurity)
        # Find states where there exists a control keeping all successors in domain
        R_sys = domain.copy()
        
        print(f"    ApplySecurity Phase 1 (system states): Initial size: {len(R_sys)}")
        
        iteration = 0
        while True:
            iteration += 1
            R_sys_new = set()
            
            for sys_state in R_sys:
                # Check if there exists a control keeping all successors in R_sys
                has_safe_control = False
                for control_idx in range(self.N_u):
                    if self._all_sys_successors_in_domain(sys_state, control_idx, R_sys):
                        has_safe_control = True
                        break
                
                if has_safe_control:
                    R_sys_new.add(sys_state)
            
            if R_sys_new == R_sys:
                break
            
            R_sys = R_sys_new
            
            if iteration % 10 == 0:
                print(f"    ApplySecurity Phase 1 iteration {iteration}: {len(R_sys)} states")
        
        print(f"    ApplySecurity Phase 1 converged after {iteration} iterations: {len(R_sys)} safe system states")
        
        # Step 2: Build product states from safe system states, excluding trash specs
        spec_automaton = self.Automaton.SpecificationAutomaton
        non_trash_specs = [s for s in spec_automaton.state_list if s not in trash_states]
        
        R = set()
        for sys_state in R_sys:
            for spec_state in non_trash_specs:
                R.add((sys_state, spec_state))
        
        print(f"    ApplySecurity Phase 2 (product states): {len(R)} states")
        
        print(f"    ApplySecurity converged after {iteration} iterations")
        return R


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
        Start controller synthesis: ApplySecurity first, then ApplyReachability.
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
        print(f"  Initial domain: {len(domain)} states")
        
        # Step 2: Apply Security - remove states that can escape to trash
        trash_states = self._get_trash_spec_states()
        print(f"  Trash specification states: {trash_states}")
        
        safe_domain = self.ApplySecurity(domain, trash_states)
        print(f"  After ApplySecurity: {len(safe_domain)} safe states")
        
        # Step 3: Apply Reachability from target states
        target_states = self._get_target_states(safe_domain)
        print(f"  Target states (accepting): {len(target_states)}")
        
        R = self.ApplyReachability(target_states, safe_domain, max_iter)
        print(f"  After ApplyReachability: {len(R)} reachable states")
        
        elapsed = time.time() - start_time
        print(f"  Controller synthesis completed in {elapsed:.2f}s")
        
        # Convert to numpy arrays for compatibility
        self._convert_h_to_arrays()
        
        # Save results
        self.Save()
        print("✓ Computed and saved new controller results to file.")
        
        return self.V, self.h_array


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
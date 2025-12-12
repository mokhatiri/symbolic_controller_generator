class Automaton:
    def __init__(self, transition, initial_state, final_states, total_states):
        self.transition = transition
        self.initial_state = initial_state
        self.final_states = final_states
        self.total_states = total_states
        
        # Build state list and mappings for efficient indexing
        self._build_state_mappings()

    def _build_state_mappings(self):
        """
        Build bidirectional mappings between state names and indices.
        This allows efficient conversion between state names and array indices.
        """
        # Extract all unique states from transitions
        all_states = set()
        
        # Add states from transition keys
        for (state, _), _ in self.transition.items():
            all_states.add(state)
        
        # Add states from transition values
        for _, next_state in self.transition.items():
            all_states.add(next_state)
        
        # Add initial and final states
        all_states.add(self.initial_state)
        all_states.update(self.final_states)
        
        # Create sorted list for consistent ordering
        self.state_list = sorted(list(all_states))
        
        # Create bidirectional mappings
        self.state_to_idx = {state: idx for idx, state in enumerate(self.state_list)}
        self.idx_to_state = {idx: state for idx, state in enumerate(self.state_list)}
        
        # Update total_states if it was set incorrectly
        if self.total_states != len(self.state_list):
            self.total_states = len(self.state_list)

    def get_next_state(self, state, input):
        """
        Get the next state given current state and input.
        
        Args:
            state: Current state (name or index)
            input: Input/label
            
        Returns:
            Next state name, or None if transition doesn't exist
        """
        # Convert index to state name if needed
        if isinstance(state, int):
            state = self.idx_to_state[state]
        
        # Return next state if transition exists
        return self.transition.get((state, input), None)
    
    def is_final_state(self, state):
        """
        Check if a state is a final/accepting state.
        
        Args:
            state: State to check (name or index)
            
        Returns:
            True if state is final, False otherwise
        """
        # Convert index to state name if needed
        if isinstance(state, int):
            state = self.idx_to_state[state]
        
        return state in self.final_states
    
    def get_state_index(self, state_name):
        """
        Get the index of a state given its name.
        
        Args:
            state_name: Name of the state
            
        Returns:
            Index of the state
        """
        return self.state_to_idx[state_name]
    
    def get_state_name(self, state_idx):
        """
        Get the name of a state given its index.
        
        Args:
            state_idx: Index of the state
            
        Returns:
            Name of the state
        """
        return self.idx_to_state[state_idx]
    
    def get_num_states(self):
        """
        Get the total number of states in the automaton.
        
        Returns:
            Number of states
        """
        return len(self.state_list)
    
    def get_outgoing_transitions(self, state):
        """
        Get all outgoing transitions from a state.
        
        Args:
            state: State to get transitions from (name or index)
            
        Returns:
            List of (input, next_state) tuples
        """
        # Convert index to state name if needed
        if isinstance(state, int):
            state = self.idx_to_state[state]
        
        outgoing = []
        for (curr_state, input_label), next_state in self.transition.items():
            if curr_state == state:
                outgoing.append((input_label, next_state))
        
        return outgoing
    
    def has_transition(self, state, input_label):
        """
        Check if a transition exists from state with given input.
        
        Args:
            state: Current state (name or index)
            input_label: Input label
            
        Returns:
            True if transition exists, False otherwise
        """
        # Convert index to state name if needed
        if isinstance(state, int):
            state = self.idx_to_state[state]
        
        return (state, input_label) in self.transition
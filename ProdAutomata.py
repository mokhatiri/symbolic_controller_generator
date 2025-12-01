class ProdAutomata:

    def __init__(self, TAutomata, SpecMap, label_map, default_states, accept_spec_states):
        self.Tmap = TAutomata.TransMap
        self.SpecMap = SpecMap
        self.label_map = label_map
        self.initial_state = default_states
        self.accept_spec_states = accept_spec_states
        self.sys_predecessor = TAutomata.predecessor
        self.accept_states = self.get_accept_states()

    def getTrans(self, state, input_symbol):
        # Combine system state and specification state
        sys_state, spec_state = state
        next_sys_states = self.Tmap.get(sys_state, {}).get(input_symbol, [])
        next_states = [(ns, self.SpecMap[spec_state][self.label_map.get(ns)]) for ns in next_sys_states]
        
        return next_states
    
    def get_accept_states(self):
        accept_states = set()
        for sys_state in self.Tmap:
            for spec_state in self.accept_spec_states:
                accept_states.add((sys_state, spec_state))
        return accept_states
    
    def predecessor_func(self, R):
        # use actualPredecessor to determine the predecessors of system states
        # supposing a definite specification automaton
        predecessors = set()

        for sys_state, spec_state in R:
            # Get actual predecessors (state, control) pairs
            sys_actual_preds = set()
            for state in self.Tmap.keys():
                for u, next_states in self.Tmap[state].items():
                    if sys_state in next_states:
                        sys_actual_preds.add((state, u))
            
            for predess_sys_state, control in sys_actual_preds:
                for predess_spec_state in self.SpecMap:
                    label = self.label_map.get(predess_sys_state, 0)
                    # Check if transitioning from predess_spec_state with label leads to spec_state
                    if self.SpecMap[predess_spec_state][label] == spec_state:
                        predecessors.add(((predess_sys_state, predess_spec_state), control))

        return predecessors
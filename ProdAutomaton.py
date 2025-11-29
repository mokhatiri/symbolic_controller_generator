class ProdAutomaton:
    def __init__(self, SpecificationAutomaton, Labeling, SymbolicAbstraction):
        self.SpecificationAutomaton = SpecificationAutomaton
        self.Labeling = Labeling
        self.SymbolicAbstraction = SymbolicAbstraction
        self.total_sys_states = self.SymbolicAbstraction.T.shape[0]
        self.total_spec_states = self.SpecificationAutomaton.states.__len__()
        self.total_states = self.total_sys_states * self.total_spec_states

    def get_labels(self, state):
        return self.Labeling[state]

    def Transition(self, next_SysState, ContextState, control_idx): # needs 
        labels = self.get_labels(next_SysState)
        possible_transitions = []
        for label in labels:
            if (ContextState, label) in self.SpecificationAutomaton.transitions:
                next_SpecState = self.SpecificationAutomaton.transitions[(ContextState, label)]
                possible_transitions.append((next_SpecState, next_SysState, control_idx))

        # in the case where the labeling is non-deterministic multiple transitions may be possible
        return possible_transitions

    def get_transitions(self, curr_state, ContextState, control_idx):
        sys_state = curr_state[1]
        successors = self.SymbolicAbstraction.T[sys_state, control_idx, :]
        all_transitions = []
        for next_sys_state in range(successors[0], successors[1] + 1):
            transitions = self.Transition(next_sys_state, ContextState, control_idx)
            all_transitions.extend(transitions)
        return all_transitions
    
    def possible_transitions(self, curr_state, ContextState):
        # check all controls
        all_transitions = []
        for control_idx in range(1, self.SymbolicAbstraction.T.shape[1] + 1):
            transitions = self.get_transitions(curr_state, ContextState, control_idx)
            all_transitions.extend(transitions)
        
        return all_transitions

    def get_initial_states(self):
        initial_states = []
        for sys_state in range(self.SylbolicAbstraction.T.shape[0]):
            labels = self.get_labels(sys_state)
            for label in labels:
                if (self.SpecificationAutomaton.initial_state, label) in self.SpecificationAutomaton.transitions:
                    next_SpecState = self.SpecificationAutomaton.transitions[(self.SpecificationAutomaton.initial_state, label)]
                    initial_states.append((next_SpecState, sys_state))
        return initial_states
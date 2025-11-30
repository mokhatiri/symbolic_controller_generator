class ProdAutomaton:
    def __init__(self, SpecificationAutomaton, Labeling, SymbolicAbstraction):
        self.SpecificationAutomaton = SpecificationAutomaton
        self.Labeling = Labeling
        self.SymbolicAbstraction = SymbolicAbstraction
        self.total_sys_states = self.SymbolicAbstraction.transition.shape[0]
        self.total_spec_states = self.SpecificationAutomaton.total_states
        self.total_states = self.total_sys_states * self.total_spec_states

    def get_labels(self, state):
        return self.Labeling[state]

    def Transition(self, next_SysState, ContextState, control_idx): # needs 
        labels = self.get_labels(next_SysState)
        possible_transition = []
        for label in labels:
            if (ContextState, label) in self.SpecificationAutomaton.transition:
                next_SpecState = self.SpecificationAutomaton.transition[(ContextState, label)]
                possible_transition.append((next_SpecState, next_SysState, control_idx))

        # in the case where the labeling is non-deterministic multiple transition may be possible
        return possible_transition

    def get_transition(self, curr_state, ContextState, control_idx):
        sys_state = curr_state[1]
        successors = self.SymbolicAbstraction.transition[sys_state, control_idx, :]
        all_transition = []
        for next_sys_state in range(successors[0], successors[1] + 1):
            transition = self.Transition(next_sys_state, ContextState, control_idx)
            all_transition.extend(transition)
        return all_transition
    
    def possible_transition(self, curr_state, ContextState):
        # check all controls
        all_transition = []
        for control_idx in range(self.SymbolicAbstraction.transition.shape[1]):
            transition = self.get_transition(curr_state, ContextState, control_idx)
            all_transition.extend(transition)
        
        return all_transition

    def get_initial_states(self):
        initial_states = []
        for sys_state in range(self.SymbolicAbstraction.transition.shape[0]):
            labels = self.get_labels(sys_state)
            for label in labels:
                if (self.SpecificationAutomaton.initial_state, label) in self.SpecificationAutomaton.transition:
                    next_SpecState = self.SpecificationAutomaton.transition[(self.SpecificationAutomaton.initial_state, label)]
                    initial_states.append((next_SpecState, sys_state))
        return initial_states
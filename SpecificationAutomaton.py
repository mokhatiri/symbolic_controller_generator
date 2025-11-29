class Automaton:

    def __init__(self, states, inputs, transitions, initial_state, final_states):
        self.states = states
        self.inputs = inputs
        self.transitions = transitions
        self.initial_state = initial_state
        self.final_states = final_states
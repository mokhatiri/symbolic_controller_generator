class Automaton:
    def __init__(self, transitions, initial_state, final_states):
        self.transitions = transitions # has the form {(state, input): next_state}
        # has the states, inputs, and transition relation
        self.initial_state = initial_state
        self.final_states = final_states

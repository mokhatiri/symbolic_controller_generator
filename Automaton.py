class Automaton:
    def __init__(self, transition, initial_state, final_states, total_states):
        self.transition = transition
        self.initial_state = initial_state
        self.final_states = final_states
        self.total_states = total_states

    def get_next_state(self, state, input):
        return self.transition[(state, input)]
    
    def is_final_state(self, state):
        return state in self.final_states
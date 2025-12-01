class Automata:
    def __init__(self, TMap, initial_state, accept_states, predecessor_func):
        self.TransMap = TMap
        self.initial_state = initial_state
        self.accept_states = accept_states
        self.predecessor = predecessor_func
        
    def get_next_states(self, state, input_symbol):
        if isinstance(self.TransMap, dict):
            return self.TransMap[state].get(input_symbol, [])
        else:
            return self.TransMap(state, input_symbol) # Assume callable
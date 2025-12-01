class Controller:
    def __init__(self, Automaton = None):
        self.Automaton = Automaton
        self.h = {}

    def setAutomaton(self, Automaton):
        self.Automaton = Automaton

    def ApplyReachability(self):
        self.h = {}
        R = set(self.Automaton.accept_states)
        while True:
            R_new = R.union(self.predecess(R))
            if R_new == R:
                break
            R = R_new

        return R

    def ApplySecurity(self):
        """
        TransMap : dict[state][alphabet] = liste de successeurs
        wrong_states : ensemble d'états interdits

        Retourne :
        - PrunedTmap : TransMap restreinte au domaine sûr R*
        """

        self.h = {}
        Qs = set(self.Automaton.non_reject_states)
        R = Qs.copy()

        while True:
            R_new = Qs.intersection(self.predecess(R))        # Qs ∩ Pre(R_k)
            if R_new == R:                                    # point fixe atteint
                break
            R = R_new

        return R
    
    def predecess(self, R):
        predecessors = self.Automaton.predecessor(R)
        for (state, control) in predecessors:
            if state not in self.h:
                self.h[state] = []
            if control not in self.h[state]:
                self.h[state].append(control)
                
        return predecessors
    
    def GetControl(self, state):
        if state in self.h:
            return self.h[state]
        else:
            return []
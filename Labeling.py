class Labeling():
    def __init__(self, states, relation, sets):
        self.states = states
        self.relation = relation
        self.sets = sets
        self.labeling_dict = self.build_labeling_dict()

    def build_labeling_dict(self):
        labeling_dict = {}
        for state in self.states:
            for label, label_set in self.sets.items():
                labeling_dict[state] = []
                if self.relation(state, label_set):
                    labeling_dict[state].append(label)

        return labeling_dict
    
    def __getitem__(self, state):
        return self.labeling_dict[state]
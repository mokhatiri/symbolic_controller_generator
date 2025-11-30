class Labeling():
    def __init__(self, states, relation, sets):
        self.states = states
        self.relation = relation
        self.sets = sets
        self.labeling_dict = self.build_labeling_dict()

    def build_labeling_dict(self):
        labeling_dict = {}
        for state in self.states:
            temp = []
            def_temp = []
            for label, label_set in self.sets.items():
                if label_set is None:
                    temp.append(label)
                else:
                    if self.relation(state, label_set):
                        def_temp.append(label)
            labeling_dict[state] = def_temp if def_temp != [] else temp

        return labeling_dict
    
    def __getitem__(self, state):
        return self.labeling_dict[state]
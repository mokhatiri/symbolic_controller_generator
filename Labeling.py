import csv
import os


class Labeling():
    def __init__(self, states, relation, sets, fname=None):
        self.states = states
        self.relation = relation
        self.sets = sets
        self.build_labeling_dict(fname=fname)

    def build_labeling_dict(self, fname=None):
        try:
            self.load_labeling_dict(fname)
        except FileNotFoundError:
            labeling_dict = {}
            for state in self.states:
                labeling_dict[state] = []
                default_label = None
                
                # Check all labels for this state
                for label, label_set in self.sets.items():
                    if label_set is None:
                        # Store default label but don't add it yet
                        default_label = label
                    else:
                        # If relation is satisfied, add this label
                        if self.relation(state, label_set):
                            labeling_dict[state].append(label)
                
                # If no labels were assigned, use the default label
                if len(labeling_dict[state]) == 0 and default_label is not None:
                    labeling_dict[state].append(default_label)

            self.labeling_dict = labeling_dict
            # save the labeling dict
            if fname is not None:
                self.save_labeling_dict(fname)
    
    def __getitem__(self, state):
        return self.labeling_dict[state]
    
    def save_labeling_dict(self, fname):
        """
        Save the labeling dictionary to a CSV file.

        Args:
            fname: Name of the CSV file to save to
        """
        # Normalize path and ensure the directory exists
        fname = os.path.normpath(fname)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        
        with open(fname, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['State', 'Labels'])
            for state, labels in self.labeling_dict.items():
                writer.writerow([state, ';'.join(map(str, labels))])

    def load_labeling_dict(self, fname):
        """
        Load the labeling dictionary from a CSV file.

        Args:
            fname: Name of the CSV file to load from
        """
        fname = os.path.normpath(fname)
        labeling_dict = {}
        with open(fname, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                state = int(row[0])
                labels = list(map(int, row[1].split(';'))) if row[1] else []
                labeling_dict[state] = labels
        self.labeling_dict = labeling_dict
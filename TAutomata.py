from collections import deque

class TAutomata:
    def __init__(self, TransMap = None):
        self.TransMap = TransMap

    def set_TransMap(self, TransMap):
        self.TransMap = TransMap

    def make_hashable(obj):
        """Recursively convert a nested structure into a hashable type."""
        if isinstance(obj, (tuple, list)):
            return tuple(TAutomata.make_hashable(e) for e in obj)
        elif isinstance(obj, dict):
            return tuple(sorted((k, TAutomata.make_hashable(v)) for k, v in obj.items()))
        return obj

    def LoadAutomataFromCsv(self, path):
        """Load SafeProd from a CSV file (type-agnostic)."""
        import csv, json

        Tmap = {}
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Deserialize everything back from JSON
                state = TAutomata.make_hashable(json.loads(row['state']))
                alphabet = TAutomata.make_hashable(json.loads(row['alphabet']))
                succ_list = TAutomata.make_hashable(json.loads(row['successors_json']))
                    
                # Successors are stored as [state, weight] pairs
                succs = [TAutomata.make_hashable(s) for s in succ_list]
                Tmap.setdefault(tuple(state), {})[alphabet] = succs

        self.TransMap = Tmap


    def SaveAutomataToCsv(self, path):
        """Save automata into a CSV file (type-agnostic)."""
        import csv, json

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['state', 'alphabet', 'successors_json'])
            for state, ctrl_dict in self.TransMap.items():
                for alphabet, succs in ctrl_dict.items():
                    # Serialize everything into JSON strings
                    writer.writerow([
                        json.dumps(state),
                        json.dumps(alphabet),
                        json.dumps(succs)
                    ])

    def ApplyReachability(self, target_states):
        R = set(target_states)
        while True:
            R_new = R.union(self.predecessor(R))
            if R_new == R:
                break
            R = R_new

        PrunedTmap = {}

        for state in R:
            Statemap = {}
            for u, next_states in self.TransMap[state].items():
                if all(ns in R for ns in next_states):
                    Statemap[u] = next_states
            if Statemap:  # Only include states that have at least one valid control
                PrunedTmap[state] = Statemap

        self.TransMap = PrunedTmap


    def ApplySecurity(self, wrong_states):
        """
        TransMap : dict[state][alphabet] = liste de successeurs
        wrong_states : ensemble d'états interdits

        Retourne :
        - PrunedTmap : TransMap restreinte au domaine sûr R*
        """

        wrong_states = set(wrong_states)

        all_states = set(self.TransMap.keys())
        for control_dict in self.TransMap.values():
            for next_states in control_dict.values():
                all_states.update(next_states)

        Qs = all_states - wrong_states

        R = Qs.copy()
        while True:
            pre = self.predecessor(R)      # Pre(R_k)
            R_new = Qs.intersection(pre)        # Qs ∩ Pre(R_k)
            if R_new == R:                      # point fixe atteint
                break
            R = R_new

        # 4) Construction de la TransMap sécurisée, restreinte à R* = R
        PrunedTmap = {}

        for state in R:
            if state not in self.TransMap:
                continue  # état sans commande définie dans TransMap
            state_map = {}
            for u, next_states in self.TransMap[state].items():
                # On ne garde que les commandes dont tous les successeurs restent dans R*
                if all(ns in R for ns in next_states):
                    state_map[u] = next_states
            if state_map:
                PrunedTmap[state] = state_map

        self.TransMap = PrunedTmap
    """
    ---------------------        UTILS:        ---------------------
    """

    def actualPredecessor(self, R):
        pred = set()
        for state, control_dict in self.TransMap.items():
            for u, next_states in control_dict.items():
                # Check if "all" next_states are in R
                if all(ns in R for ns in next_states):
                    pred.add((state, u))
        return pred
    
    def predecessor(self, R):
        actual_pred = self.actualPredecessor(R)
        return set(state for state, u in actual_pred)

    def bfs_trajectory(self, startStates, goalStates):
        """
        BFS to find a trajectory from startStates to any of the goalStates.

        TransMap : dict[x][u] -> list[x_next]
        startStates : list of starting states ((i,j,k),'state')
        goalStates : set of goal states

        Returns :
            path : list of x pairs leading from start to goal, or None if no path
        """
        queue = deque()
        visited = set()

        # Initialize BFS with start states
        for s in startStates:
            queue.append((s, []))  # (current_state, path_so_far)
            visited.add(s)
            

        while queue:
            state, path = queue.popleft()

            if state in goalStates:
                return path+[state]  # reached a goal

            if state not in self.TransMap:
                continue

            for u, next_states in self.TransMap[state].items():
                for next_state in next_states:
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append((next_state, path + [state]))

        return None
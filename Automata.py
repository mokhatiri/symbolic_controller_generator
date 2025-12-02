from collections import deque

class Automata:
    def __init__(self, TransMap = None):
        self.TransMap = TransMap
        self.h={}

    def set_TransMap(self, TransMap):
        self.TransMap = TransMap

    def make_hashable(obj):
        """Recursively convert a nested structure into a hashable type."""
        if isinstance(obj, (tuple, list)):
            return tuple(Automata.make_hashable(e) for e in obj)
        elif isinstance(obj, dict):
            return tuple(sorted((k, Automata.make_hashable(v)) for k, v in obj.items()))
        return obj

    def LoadAutomataFromCsv(self, path):
        """Load SafeProd from a CSV file (type-agnostic)."""
        import csv, json

        Tmap = {}
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Deserialize everything back from JSON
                state = Automata.make_hashable(json.loads(row['state']))
                alphabet = Automata.make_hashable(json.loads(row['alphabet']))
                succ_list = Automata.make_hashable(json.loads(row['successors_json']))
                    
                # Successors are stored as [state, weight] pairs
                succs = [Automata.make_hashable(s) for s in succ_list]
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
        self.h = {}
        R = set(target_states)
        while True:
            R_new = R.union(self.predecessor(R))
            if R_new == R:
                break
            R = R_new

        return R

    def ApplySecurity(self, correct_states):
        """
        TransMap : dict[state][alphabet] = liste de successeurs
        correct_states : ensemble d'états interdits

        Retourne :
        - PrunedTmap : TransMap restreinte au domaine sûr R*
        """
        Qs = correct_states

        R = Qs.copy()
        print("R: ", R)
        count = 0
        count1 = 0
        while True:
            pred = set()
            for state, control_dict in self.TransMap.items():
                for u, next_states in control_dict.items():
                    # Check if "all" next_states are in R
                    if all(ns in R for ns in next_states) and next_states:
                        pred.add(state)
                        count1 += 1
                    else:
                        count += 1

            R_new = Qs.intersection(pred)       # Qs ∩ Pre(R_k) # leave only the states within the bounds
            print(R_new, count, count1)
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
                if all(ns in R for ns in next_states):
                    state_map[u] = next_states
            if state_map:
                PrunedTmap[state] = state_map

        self.TransMap = PrunedTmap

        print(R)
        return R

    from collections import deque

    def ToProdAutomate(self, SpecMap, LabelMap, start_SpecMap):
        """
        Build the reachable product automaton starting from start_SpecMap.
        
        Only include states (x, q) that are reachable from the given starting specification state.
        
        TransMap   : dict[state][u] -> list[next_state]
        SpecMap    : dict[q][label] -> next_spec_state
        LabelMap   : dict[state] -> label
        start_SpecMap : initial specification state (default 'a')
        
        Returns:
            ProdMap[(x,q)][u] = list of (x_next, q_next)
        """
        ProdMap = {}
        visited = set()
        queue = deque()

        # Initialize BFS with all system states at start_SpecMap
        for x in self.TransMap.keys():
            prod_state = (x, start_SpecMap)
            queue.append(prod_state)
            visited.add(prod_state)

        while queue:
            current = queue.popleft()
            x, q = current
            if x not in self.TransMap:
                continue

            state_transitions = {}
            for u, next_states in self.TransMap[x].items():
                prod_succs = []
                for x_next in next_states:
                    if (x_next[0],x_next[1]) not in LabelMap:
                        continue
                    label = LabelMap[(x_next[0],x_next[1])]
                    if q not in SpecMap or label not in SpecMap[q]:
                        continue
                    q_next = SpecMap[q][label]
                    succ_state = (x_next, q_next)
                    prod_succs.append(succ_state)

                    if succ_state not in visited:
                        visited.add(succ_state)
                        queue.append(succ_state)

                if prod_succs:
                    state_transitions[u] = prod_succs

            if state_transitions:
                ProdMap[current] = state_transitions
        
        self.TransMap = ProdMap

    """
    ---------------------        UTILS:        ---------------------
    """

    def predecessor(self, R):
        pred = set()
        for state, control_dict in self.TransMap.items():
            for u, next_states in control_dict.items():
                # Check if "all" next_states are in R
                if all(ns in R for ns in next_states) and next_states:
                    pred.add(state)
                    if not state in self.h:
                        print("Setting h[{}] = {}".format(state, u))
                        self.h[state] = u # select the first
        return pred
    
    def getH(self):
        return self.h

    def bfs_trajectory(self, startStates, goalStates):
        """
        BFS to find a trajectory from startStates to any of the goalStates,
        using only controls from the controller h.

        TransMap : dict[x][u] -> list[x_next]
        startStates : list of starting states ((i,j,k),'state')
        goalStates : set of goal states

        Returns :
            path : list of states leading from start to goal, or None if no path
        """
        queue = deque()
        visited = set()

        # Initialize BFS with start states
        for s in startStates:
            queue.append((s, [s]))  # (current_state, path_including_current)
            visited.add(s)

        while queue:
            state, path = queue.popleft()

            if state in goalStates:
                return path  # reached a goal

            # Only use controls from the controller h
            if state not in self.h:
                continue
            
            # h[state] can be a single control or a list of controls
            controls = self.h[state] if isinstance(self.h[state], list) else [self.h[state]]
            
            for u in controls:
                if state not in self.TransMap or u not in self.TransMap[state]:
                    continue
                    
                for next_state in self.TransMap[state][u]:
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append((next_state, path + [next_state]))

        return None
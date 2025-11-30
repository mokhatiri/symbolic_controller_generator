class ProdAutomaton:
    def __init__(self, SpecificationAutomaton, Labeling, SymbolicAbstraction):
        self.SpecificationAutomaton = SpecificationAutomaton
        self.Labeling = Labeling
        self.SymbolicAbstraction = SymbolicAbstraction
        self.total_sys_states = self.SymbolicAbstraction.transition.shape[0]
        self.total_spec_states = self.SpecificationAutomaton.total_states
        self.total_states = self.total_sys_states * self.total_spec_states
        
        # OPTIMIZATION: Cache transitions with bounded size to prevent OOM
        # Only keep most recent 50,000 queries (LRU eviction on overflow)
        self._transition_cache = {}
        self._cache_order = []  # Track insertion order for LRU
        self.MAX_CACHE_SIZE = 50000  # Prevent unbounded memory growth
        self._cache_hits = 0
        self._cache_misses = 0

    def get_labels(self, state):
        return self.Labeling[state]

    def Transition(self, next_SysState, ContextState, control_idx): # needs 
        labels = self.get_labels(next_SysState)
        possible_transition = []
        for label in labels:
            if (ContextState, label) in self.SpecificationAutomaton.transition:
                next_SpecState = self.SpecificationAutomaton.transition[(ContextState, label)]
                possible_transition.append((next_SpecState, next_SysState, control_idx))

        # in the case where the labeling is non-deterministic multiple transition may be possible
        return possible_transition

    def get_transition(self, curr_state, ContextState, control_idx):
        """
        Get successors for a given state and control.
        
        **OPTIMIZATION v3**: Cache transitions with bounded size (LRU eviction)
        to avoid recomputation while preventing OOM on large systems.
        This is especially important for fixed-point iterations that query the same
        transitions multiple times.
        
        **Cache Management**: Only keeps 50,000 most recent queries. Older entries
        are evicted to prevent unbounded memory growth on large product automata.
        """
        # Create cache key
        cache_key = (curr_state, ContextState, control_idx)
        
        if cache_key in self._transition_cache:
            self._cache_hits += 1
            # Move to end (mark as recently used)
            if cache_key in self._cache_order:
                self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            return self._transition_cache[cache_key]
        
        self._cache_misses += 1
        
        sys_state = curr_state[1]
        successors = self.SymbolicAbstraction.transition[sys_state, control_idx, :]
        all_transition = []
        for next_sys_state in range(successors[0], successors[1] + 1):
            transition = self.Transition(next_sys_state, ContextState, control_idx)
            all_transition.extend(transition)
        
        # Cache result
        self._transition_cache[cache_key] = all_transition
        self._cache_order.append(cache_key)
        
        # Evict oldest entry if cache exceeds max size (LRU)
        if len(self._transition_cache) > self.MAX_CACHE_SIZE:
            oldest_key = self._cache_order.pop(0)
            del self._transition_cache[oldest_key]
        
        return all_transition
    
    def possible_transition(self, curr_state, ContextState):
        # check all controls
        all_transition = []
        for control_idx in range(self.SymbolicAbstraction.transition.shape[1]):
            transition = self.get_transition(curr_state, ContextState, control_idx)
            all_transition.extend(transition)
        
        return all_transition

    def get_initial_states(self):
        initial_states = []
        for sys_state in range(self.SymbolicAbstraction.transition.shape[0]):
            labels = self.get_labels(sys_state)
            for label in labels:
                if (self.SpecificationAutomaton.initial_state, label) in self.SpecificationAutomaton.transition:
                    next_SpecState = self.SpecificationAutomaton.transition[(self.SpecificationAutomaton.initial_state, label)]
                    initial_states.append((next_SpecState, sys_state))
        return initial_states
    
    def print_cache_stats(self):
        """Print cache performance statistics (OPTIMIZATION)."""
        total_queries = self._cache_hits + self._cache_misses
        hit_rate = 100 * self._cache_hits / total_queries if total_queries > 0 else 0
        cache_size_mb = sum(len(str(v)) for v in self._transition_cache.values()) / 1024 / 1024
        print(f"Transition cache stats: {self._cache_hits} hits, {self._cache_misses} misses, {hit_rate:.1f}% hit rate")
        print(f"Cache size: {len(self._transition_cache)}/{self.MAX_CACHE_SIZE} entries, ~{cache_size_mb:.1f} MB")
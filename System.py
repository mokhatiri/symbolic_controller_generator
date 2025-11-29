class System:
    def __init__(self, f, Jx, Jw):
        self.f = f  # System dynamics function
        self.Jx = Jx  # Jacobian % state
        self.Jw = Jw  # Jacobian % disturbance
        self.curr_x = None  # Current state

    def set_initial_state(self, x0):
        self.curr_x = x0
    
    def step(self, u, w):
        """
        Advance the system state by one time step using the dynamics function.
        
        Args:
            u: Control input
            w: Disturbance input
            
        Returns:
            Updated state after applying control and disturbance
        """
        self.curr_x = self.f(self.curr_x, u, w)
        return self.curr_x
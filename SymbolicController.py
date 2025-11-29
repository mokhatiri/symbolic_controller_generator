import Discretisation
import Labeling
import System
import AbstractSpace
import ProdAutomaton
import Controller

class SymbolicController:
 
    def __init__(self, f, Jx, Jw, X_bounds, U_bounds, W_bounds, cells_per_dim_x, cells_per_dim_u, angular_dims_x, SpecificationAutomaton, states, relation, sets):
        """
        Initialize the symbolic controller.
        """
        self.u = None
        self.context = SpecificationAutomaton.initial_state # is an Automaton

        self.Discretisation = Discretisation(X_bounds, U_bounds, W_bounds, cells_per_dim_x, cells_per_dim_u, angular_dims_x)
        self.System = System(f, Jx, Jw)

        labeling = Labeling(states, relation, sets)
        abstraction = AbstractSpace(self.System, self.Discretisation)
        Automaton = ProdAutomaton(abstraction, SpecificationAutomaton, labeling)

        self.Controller = Controller(Automaton)

    def start(self, x0, context, is_reachability=True, max_iter=10000):
        """
        Start the controller with the initial state and context.

        Args:
            x0: Initial state
            context: Context information
        """
        self.x = x0
        self.u = None
        self.context = context

        self.V, self.h = self.Controller.Start(is_reachability, max_iter)

    def step(self, disturbance):
        """
        Advance the controller by one time step using the disturbance input.

        Args:
            disturbance: Disturbance input

        Returns:
            Control input for the current state
        """

        # step 1: x+ = f(x, u, disturbance)
        if self.u is not None:
            self.x = self.step(self.u, disturbance)

        # step 2: get discrete state index
        state_idx = self.Discretisation.coord_to_idx(self.Discretisation.continuous_to_cell_idx(self.System.curr_x))
    
        # step 3: get control input index from controller
        control_idx = self.h[(state_idx, self.context)]

        # step 4: convert control index to continuous control input
        control_input = self.Discretisation.control_cell_idx_to_continuous(
            self.Discretisation.idx_to_coord(control_idx)
        )
        self.u = control_input

        return control_input
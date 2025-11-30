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
        self.x = None

        self.Discretisation = Discretisation.Discretisation(X_bounds, U_bounds, W_bounds, cells_per_dim_x, cells_per_dim_u, angular_dims_x)
        self.System = System.System(f, Jx, Jw)

        labeling = Labeling.Labeling(states, relation, sets)
        abstraction = AbstractSpace.AbstractSpace(self.System, self.Discretisation)
        automaton = ProdAutomaton.ProdAutomaton(abstraction, SpecificationAutomaton, labeling)

        self.Controller = Controller.ControllerSynthesis(automaton)
        self.V = None
        self.h = None

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
            self.x = self.System.step(self.u, disturbance)

        # step 2: get discrete state index
        continuous_cell_idx = self.Discretisation.continuous_to_cell_idx(self.x)
        state_idx = self.Discretisation.coord_to_idx(continuous_cell_idx)
    
        # step 3: get control input index from controller
        product_state_idx = state_idx  # Product state combines (spec_state, sys_state)
        control_idx = self.h[product_state_idx]

        # step 4: convert control index to continuous control input
        if control_idx != -1:
            control_cell_idx = self.Discretisation.idx_to_coord(control_idx)
            control_input = self.Discretisation.idx_to_continuous(control_cell_idx, 
                                                                   self.Discretisation.U_bounds, 
                                                                   self.Discretisation.cells_per_dim_u)
        else:
            control_input = None
            
        self.u = control_input

        return control_input
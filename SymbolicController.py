import Discretisation
import Labeling
import System
import AbstractSpace
import ProdAutomaton
import Controller

class SymbolicController:
 
    def __init__(self, f, Jx, Jw, X_bounds, U_bounds, W_bounds, cells_per_dim_x, cells_per_dim_u, angular_dims_x, SpecificationAutomaton, relation, sets, model_dir='./Models'):
        """
        Initialize the symbolic controller.
        
        Args:
            model_dir: Directory to save/load model files (default: './Models')
        """
        self.u = None
        self.context = SpecificationAutomaton.initial_state # is an Automaton
        self.x = None
        self.model_dir = model_dir

        self.Discretisation = Discretisation.Discretisation(X_bounds, U_bounds, W_bounds, cells_per_dim_x, cells_per_dim_u, angular_dims_x)
        self.System = System.System(f, Jx, Jw)

        def actual_relation(state_idx, set):
            max_state , min_state = self.Discretisation.idx_to_state_bounds(state_idx)
            return relation(min_state, max_state, set)

        # Pass state indices (0 to N_x-1) instead of continuous state values
        state_indices = list(range(self.Discretisation.N_x))
        labeling = Labeling.Labeling(state_indices, actual_relation, sets)
        abstraction = AbstractSpace.AbstractSpace(self.System, self.Discretisation, model_dir=model_dir)
        automaton = ProdAutomaton.ProdAutomaton(SpecificationAutomaton, labeling, abstraction)

        self.Controller = Controller.ControllerSynthesis(automaton, model_dir=model_dir)
        self.V = None
        self.h = None

    def start(self, is_reachability=True, max_iter=10000):
        """
        Start the controller with the initial state and context.

        Args:
            x0: Initial state
            context: Context information
        """
        self.V, self.h = self.Controller.Start(is_reachability, max_iter)


    def goto(self, x0):
        """
        Set the continuous state x.
                
        Args:
            x0: continuous state
        """
        self.x = x0

    def setContext(self, context):
        """
        Set the context of the controller.

        Args:
            context: New context
        """
        self.context = context

    def stepfrom(self, x0, context, disturbance):
        """
        Advance the controller by one time step from a given state and context.

        Args:
            x0: Continuous state
            context: Context information
        """
        self.goto(x0)
        self.setContext(context)

        return self.step(disturbance)

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
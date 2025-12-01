import Discretisation
import Labeling
import System
import AbstractSpace
import ProdAutomaton
import Controller
import time
from datetime import datetime

class SymbolicController:
 
    def __init__(self, f, Jx, Jw, X_bounds, U_bounds, W_bounds, cells_per_dim_x, cells_per_dim_u, angular_dims_x, SpecificationAutomaton, relation, sets, model_dir='./Models'):
        """
        Initialize the symbolic controller with progress tracking.
        
        Args:
            model_dir: Directory to save/load model files (default: './Models')
        """
        init_start_time = time.time()
        print(f"\n{'='*70}")
        print(f"🚀 Initializing Symbolic Controller @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        self.u = None
        self.context = SpecificationAutomaton.initial_state # is an Automaton
        self.x = None
        self.model_dir = model_dir

        # Step 1: Discretisation
        step1_start = time.time()
        print(f"\n[1/5] Creating discretisation...")
        self.Discretisation = Discretisation.Discretisation(X_bounds, U_bounds, W_bounds, cells_per_dim_x, cells_per_dim_u, angular_dims_x)
        step1_time = time.time() - step1_start
        print(f"      ✓ Discretisation complete ({step1_time:.2f}s)")
        print(f"        • State space: {self.Discretisation.N_x:,} cells")
        print(f"        • Control space: {self.Discretisation.N_u:,} cells")
        
        # Step 2: System
        step2_start = time.time()
        print(f"\n[2/5] Creating system dynamics...")
        self.System = System.System(f, Jx, Jw)
        step2_time = time.time() - step2_start
        print(f"      ✓ System created ({step2_time:.2f}s)")

        # Step 3: Labeling
        step3_start = time.time()
        print(f"\n[3/5] Computing state labeling...")
        def actual_relation(state_idx, set):
            max_state , min_state = self.Discretisation.idx_to_state_bounds(state_idx)
            return relation(min_state, max_state, set)

        # Pass state indices (0 to N_x-1) instead of continuous state values
        state_indices = list(range(self.Discretisation.N_x))
        labeling = Labeling.Labeling(state_indices, actual_relation, sets)
        step3_time = time.time() - step3_start
        print(f"      ✓ Labeling complete ({step3_time:.2f}s)")
        print(f"        • Labeled {self.Discretisation.N_x:,} states")

        # Step 4: Symbolic Abstraction
        step4_start = time.time()
        print(f"\n[4/5] Computing symbolic abstraction...")
        abstraction = AbstractSpace.AbstractSpace(self.System, self.Discretisation, model_dir=model_dir)
        step4_time = time.time() - step4_start
        print(f"      ✓ Abstraction complete ({step4_time:.2f}s)")

        # Step 5: Product Automaton & Controller
        step5_start = time.time()
        print(f"\n[5/5] Assembling product automaton...")
        automaton = ProdAutomaton.ProdAutomaton(SpecificationAutomaton, labeling, abstraction)
        print(f"      ✓ Product automaton created")
        print(f"        • System states: {automaton.total_sys_states:,}")
        print(f"        • Spec states: {automaton.total_spec_states}")
        print(f"        • Product states: {automaton.total_states:,}")
        
        self.Controller = Controller.ControllerSynthesis(automaton, model_dir=model_dir)
        step5_time = time.time() - step5_start
        print(f"      ✓ Controller created ({step5_time:.2f}s)")

        total_init_time = time.time() - init_start_time
        print(f"\n{'='*70}")
        print(f"✅ Initialization complete in {total_init_time:.2f}s")
        print(f"{'='*70}\n")
        
        self.V = None
        self.h = None

    def start(self, is_reachability=True, max_iter=10000):
        """
        Start the controller synthesis with progress tracking.

        Args:
            is_reachability: If True, solve reachability. If False, solve safety.
            max_iter: Maximum number of iterations for fixed-point computation
        """
        synthesis_start_time = time.time()
        mode = "Reachability" if is_reachability else "Safety"
        
        print(f"\n{'='*70}")
        print(f"🎯 Starting Controller Synthesis ({mode}) @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        print(f"Max iterations: {max_iter:,}")
        print(f"Problem type: {mode} Problem")
        print(f"{'='*70}\n")
        
        self.V, self.h = self.Controller.Start(is_reachability, max_iter)
        
        synthesis_time = time.time() - synthesis_start_time
        
        # Print synthesis summary
        valid_controls = int((self.h >= 0).sum())
        total_states = len(self.h)
        controllable_pct = 100 * valid_controls / total_states if total_states > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"✅ Synthesis complete in {synthesis_time:.2f}s")
        print(f"{'='*70}")
        print(f"Results:")
        print(f"  • Valid control inputs: {valid_controls:,} / {total_states:,} ({controllable_pct:.1f}%)")
        if is_reachability:
            reachable = int((self.V >= 0).sum())
            print(f"  • Reachable states: {reachable:,} / {total_states:,}")
            if reachable > 0:
                print(f"  • Min steps to target: {self.V[self.V >= 0].min()}")
                print(f"  • Max steps to target: {self.V[self.V >= 0].max()}")
        else:
            safe = int((self.V > 0).sum())
            print(f"  • Safe states: {safe:,} / {total_states:,}")
        print(f"{'='*70}\n")


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
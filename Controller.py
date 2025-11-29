import numpy as np
import time
from Labeling import Labeling
from Discretisation import Abstract
from System import System

class SymbolicController:

    def __init__(self, SymbolicAbstraction, Labeling, Discretisation):
        """
        Initialize the symbolic controller.
        
        Args:
            SymbolicAbstraction: Instance of the symbolic abstraction (transition system)
            Labeling: Instance of the labeling function
            Discretisation: Instance of discretisation parameters
        """
        self.SymbolicAbstraction = SymbolicAbstraction
        self.Labeling = Labeling
        self.Discretisation = Discretisation

        # Choice of a successor sample 
        self.gSample = np.zeros((self.Discretisation.N_x, self.Discretisation.N_x, self.Discretisation.N_u), dtype=int)
        for psi in range(self.Discretisation.N_x):
            self.gSample[psi, :, :] = SymbolicAbstraction[:, :, 0]

        # Initialize value function and controller
        self.V = np.inf * np.ones((self.Discretisation.N_x, self.Discretisation.N_x))
        self.V5 = np.zeros((self.Discretisation.N_x, self.Discretisation.N_x), dtype=int)
        # TODO: Set initial states from specification (F_s should come from specification)
        # for f in F_s:
        #     self.V[f - 1, :] = 0
        
        self.h2 = np.zeros((self.Discretisation.N_x, self.Discretisation.N_x), dtype=int)
    
    def Start(self, max_iter=100):
        """
        Start the controller synthesis process.
        
        Args:
            max_iter: Maximum number of iterations for fixed-point computation
            
        Returns:
            Value function and controller mapping
        """
        self.h25 = np.zeros((self.Discretisation.N_x, self.Discretisation.N_x), dtype=int)
        return self.Load().SynthesisController(max_iter)
    
    def SynthesisController(self, max_iter):
        """
        Synthesize the controller using fixed-point iteration.
        
        Implements the fixed-point algorithm to compute the largest set of states
        from which the specification can be satisfied.
        
        Args:
            max_iter: Maximum number of iterations
            
        Returns:
            Tuple of (value function V, controller h2)
        """
        # Fixed-point iteration
        start_time = time.time()
        for iter in range(max_iter):
            Vp = self.V.copy()
            
            for xi in range(self.Discretisation.N_x):
                for psi in range(1, self.Discretisation.N_x + 1):
                    # TODO: psiSucc computation depends on labeling function h1
                    # psiSucc = h1(psi, xi, self.Labeling)

                    if self.V[psi - 1, xi] == np.inf:
                        if self.h2[psiSucc - 1, xi] != 0:
                            self.V5[psi - 1, xi] = iter 
                        else:
                            Vmax = np.zeros(self.Discretisation.N_u)
                            for sigma in range(self.Discretisation.N_u):
                                xi_succ = self.SymbolicAbstraction[xi, sigma, :]
                                if np.all(xi_succ):
                                    if Vp[psiSucc - 1, self.gSample[psi - 1, xi, sigma] - 1] != np.inf:
                                        # TODO: Implement state coordinate conversion based on Discretisation
                                        # cMin = self.Discretisation.idx_to_coord(int(xi_succ[0]))
                                        # cMax = self.Discretisation.idx_to_coord(int(xi_succ[1]))
                                        pass

                                        # if cMin[2] <= cMax[2]: 
                                        #     qSucc = self.Discretisation.coord_to_idx(cMin), ...
                                        # else:
                                        #     qSucc = ...
                                        
                                        # vSucc = Vp[psiSucc - 1, qSucc - 1]
                                        # Vmax[sigma] = np.all(vSucc != np.inf) 

                                        # if Vmax[sigma]:
                                        #     break 
                                        # else: 
                                        #     i_p = np.argmax(vSucc != np.inf)
                                        #     self.gSample[psi - 1, xi, sigma] = qSucc[i_p]
                        
                            if np.any(Vmax):
                                self.V5[psi - 1, xi] = iter 
                                sigma = np.max(Vmax)
                                self.h25[psiSucc - 1, xi] = sigma + 1

            
            # Check convergence
            elapsed_time = time.time() - start_time
            if np.array_equal(Vp, self.V) and elapsed_time > 600:  # timeout condition (divergence)
                print("Fixed-point algorithm reached convergence.")
                break
        
        controllerCount = np.count_nonzero(self.h2)
        totalStates = self.h2.size
        print(f"\nController Coverage: {controllerCount}/{totalStates} states")
        
        return self.V, self.h2
    

    # Saving and loading for time efficiency (avoiding recomputation)
    def Load(self):
        """
        Load pre-computed value function and controller from file.
        
        Returns:
            self (for method chaining)
        """
        try:
            self.V = np.loadtxt('./Models/V_result.csv', delimiter=',')
            self.h2 = np.loadtxt('./Models/h2_result.csv', delimiter=',')
        except FileNotFoundError:
            print("No saved results found. Starting fresh computation.")
        return self

    def Save(self):
        """
        Save the computed value function and controller to file.
        """
        np.savetxt('./Models/V_result.csv', self.V, delimiter=',')
        np.savetxt('./Models/h2_result.csv', self.h2, delimiter=',')
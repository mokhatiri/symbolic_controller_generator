# Symbolic Controller Generator

A Python framework for synthesizing discrete controllers for continuous nonlinear systems using symbolic abstraction and automata-based specifications.

## Overview

This framework implements the symbolic controller synthesis methodology described in the accompanying research paper. It enables the automatic generation of discrete controllers that guarantee satisfaction of temporal specifications while accounting for system dynamics and bounded disturbances.

### Key Concepts

- **Symbolic Abstraction**: Maps a continuous nonlinear system to a finite discrete model
- **Product Automaton**: Combines system dynamics with a specification automaton
- **Controller Synthesis**: Computes a control policy satisfying both reachability and safety objectives
- **Discretization**: Uses interval arithmetic with Jacobians to compute reachable sets

---

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd symbolic_controller_generator
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

The following packages are required:
- `numpy>=1.20.0` - Numerical computations
- `pandas>=1.3.0` - Data handling
- `matplotlib>=3.4.0` - Visualization (optional)

---

## Quick Start

### Basic Usage Example

```python
import numpy as np
from SymbolicController import SymbolicController
from Automaton import Automaton

# Step 1: Define system dynamics
def system_dynamics(x, u, w):
    """Continuous system: x_{k+1} = f(x_k, u_k, w_k)"""
    return 0.9 * x + 0.1 * u + w

def jacobian_x(u):
    """Jacobian with respect to state"""
    return 0.9

def jacobian_w(u):
    """Jacobian with respect to disturbance"""
    return 1.0

# Step 2: Define bounds
X_bounds = np.array([[-2.0, 2.0]])  # State space
U_bounds = np.array([[-1.0, 1.0]])  # Control input
W_bounds = np.array([[-0.2, 0.2]])  # Disturbance

# Step 3: Define discretization
cells_per_dim_x = np.array([5])
cells_per_dim_u = np.array([3])

# Step 4: Define specification automaton
def transition_func(state, label):
    if state == 0 and label == 'target':
        return 1
    return state

spec_automaton = Automaton(
    transition=transition_func,
    initial_state=0,
    final_states=[1],
    total_states=2
)

# Step 5: Initialize symbolic controller
controller = SymbolicController(
    f=system_dynamics,
    Jx=jacobian_x,
    Jw=jacobian_w,
    X_bounds=X_bounds,
    U_bounds=U_bounds,
    W_bounds=W_bounds,
    cells_per_dim_x=cells_per_dim_x,
    cells_per_dim_u=cells_per_dim_u,
    angular_dims_x=[],
    SpecificationAutomaton=spec_automaton,
    states=list(range(5)),
    relation=lambda s, ls: True,  # State labeling function
    sets={'target': {}}
)

# Step 6: Synthesize controller
x0 = np.array([0.0])
controller.start(x0, context=0, is_reachability=True, max_iter=10000)

# Step 7: Use controller in simulation
x = x0.copy()
for step in range(20):
    u = controller.step(disturbance=0.0)
    if u is None:
        print(f"No control available at step {step}")
        break
    print(f"Step {step}: x={x[0]:.3f}, u={u:.3f}")
```

---

## Core Components

### 1. SymbolicController Class

**Main interface** for synthesizing and using controllers.

#### Constructor

```python
SymbolicController(
    f: callable,                          # System dynamics function f(x, u, w)
    Jx: callable,                         # Jacobian ∂f/∂x
    Jw: callable,                         # Jacobian ∂f/∂w
    X_bounds: np.ndarray,                 # State bounds [n_dims, 2]
    U_bounds: np.ndarray,                 # Control bounds [n_ctrl, 2]
    W_bounds: np.ndarray,                 # Disturbance bounds [n_dist, 2]
    cells_per_dim_x: np.ndarray,          # Discretization cells per state dimension
    cells_per_dim_u: np.ndarray,          # Discretization cells per control dimension
    angular_dims_x: list,                 # Indices of angular dimensions (0-indexed)
    SpecificationAutomaton: Automaton,    # Specification automaton
    states: list,                         # List of discrete states
    relation: callable,                   # Labeling relation function
    sets: dict                            # Labeled sets
)
```

#### Methods

##### `start(x0, context, is_reachability=True, max_iter=10000)`

Synthesizes the controller using fixed-point iteration.

**Parameters:**
- `x0` (np.ndarray): Initial continuous state
- `context` (int): Initial specification automaton state
- `is_reachability` (bool): If True, solve reachability; if False, solve safety
- `max_iter` (int): Maximum iterations for fixed-point computation

**Returns:**
- Modifies internal state (`self.V`, `self.h`)
- `V`: Value function array
- `h`: Control policy array

**Example:**
```python
controller.start(
    x0=np.array([0.0, 0.0]),
    context=0,
    is_reachability=True,
    max_iter=10000
)

# Access results
value_function = controller.V  # steps to target or safety status
control_policy = controller.h  # optimal control at each state
```

##### `step(disturbance)`

Executes one control step in closed-loop.

**Parameters:**
- `disturbance` (float or np.ndarray): Current disturbance input

**Returns:**
- `control_input` (float or np.ndarray): Continuous control to apply, or None if no valid control

**Example:**
```python
for t in range(100):
    u = controller.step(disturbance=0.1)
    if u is None:
        print("Controller failed: unreachable state")
        break
    x_next = apply_control(u)  # Apply to real system
```

### 2. Discretisation Class

Handles state and control space discretization with interval arithmetic.

**Key Methods:**
- `continuous_to_cell_idx(x)`: Convert continuous state to cell coordinates
- `coord_to_idx(coord)`: Convert cell coordinates to linear state index
- `idx_to_continuous(idx)`: Convert state index back to continuous coordinates
- `discretize_control()`: Generate all discrete control inputs
- `discretize_state()`: Generate all discrete state centers

### 3. AbstractSpace Class

Computes symbolic abstraction through reachable set computation.

**Key Methods:**
- `compute_symbolic_model()`: Builds transition relation T
- `compute_reachable_set(x, u, w)`: Computes reachable set interval using Jacobians
- `normalize_angular_bounds(R)`: Wraps angular dimensions to [-π, π]

### 4. Controller Synthesis (ControllerSynthesis Class)

Synthesizes controller policies using fixed-point iteration.

**Methods:**
- `Start(is_reachability, max_iter)`: Main synthesis entry point
- `SynthesisReachabilityController(max_iter)`: Backwards reachability iteration
- `SynthesisSafetyController(max_iter)`: Forward safety iteration

---

## Running Examples

### Example 1: Double Integrator System

A classic control problem (position and velocity control).

```bash
python example_double_integrator.py
```

**Output:**
- Synthesis results (reachable states, value function statistics)
- Closed-loop simulation trajectory
- Step-by-step control sequence

### Example 2: 2D Nonlinear System

A more complex example with coupled dynamics.

```bash
python example.py
```

**Features:**
- Multivariate system dynamics
- Specification automaton
- Detailed results analysis

### Creating Your Own Example

Use `example_template.py` as a starting point:

```bash
cp example_template.py my_example.py
# Edit my_example.py with your system
python my_example.py
```

---

## System Dynamics Definition

### Required Components

When defining a system, you must provide three components:

#### 1. Dynamics Function

```python
def dynamics(x, u, w):
    """
    Continuous system dynamics: x_{k+1} = f(x_k, u_k, w_k)
    
    Args:
        x: Current state (np.ndarray of shape [n_x])
        u: Control input (np.ndarray of shape [n_u])
        w: Disturbance (np.ndarray or float)
    
    Returns:
        x_next: Next state (np.ndarray of shape [n_x])
    """
    A = np.array([[0.9, 0.1],
                  [0.1, 0.8]])
    B = np.array([0.5, 0.3])
    return A @ x + B * u + w
```

#### 2. State Jacobian

```python
def jacobian_state(u):
    """
    Jacobian matrix ∂f/∂x(x, u, w)
    
    Returns:
        J_x: Jacobian (np.ndarray of shape [n_x, n_x])
    """
    return np.array([[0.9, 0.1],
                     [0.1, 0.8]])
```

#### 3. Disturbance Jacobian

```python
def jacobian_disturbance(u):
    """
    Jacobian matrix ∂f/∂w(x, u, w)
    
    Returns:
        J_w: Jacobian (np.ndarray of shape [n_x, n_w])
    """
    return np.array([1.0, 1.0])
```

### Tips

- **Linear Systems**: Jacobians are constant matrices
- **Nonlinear Systems**: Jacobians can depend on u or x (evaluate at nominal point)
- **Worst-case bounds**: Use maximum magnitude entries in Jacobians for conservative abstraction

---

## Specification Definition

### Labeling Function

Maps discrete states to labels based on membership in labeled sets:

```python
def state_relation(state_idx, label_set):
    """
    Determine if state belongs to a labeled region.
    
    Args:
        state_idx: Discrete state index
        label_set: Dictionary with region information
    
    Returns:
        bool: True if state is in region, False otherwise
    """
    if 'bounds' in label_set:
        lower, upper = label_set['bounds']
        state_coords = discretisation.idx_to_coord(state_idx)
        return np.all(lower <= state_coords) and np.all(state_coords <= upper)
    return False
```

### Specification Automaton

Define as an `Automaton` object:

```python
def spec_transition(state, label):
    """Specification transition function (state, label) -> next_state"""
    transition = {
        (0, 'reach_target'): 1,
        (1, 'safe'): 1,
    }
    return transition.get((state, label))

spec = Automaton(
    transition=spec_transition,
    initial_state=0,           # Start state
    final_states=[1],          # Accepting states
    total_states=2
)
```

---

## Output Interpretation

### Value Function V

**For Reachability:**
- `V[state] = k`: Minimum k steps to reach target from state
- `V[state] = -1`: Unreachable state

**For Safety:**
- `V[state] = 1`: State is safely controllable
- `V[state] = 0`: Unsafe state

### Control Policy h

- `h[state] = control_idx`: Optimal control index at state
- `h[state] = -1`: No valid control (unreachable/unsafe state)

### Example Analysis

```python
# Print reachability statistics
reachable_count = np.sum(controller.V >= 0)
total_states = len(controller.V)
print(f"Reachable: {reachable_count}/{total_states} ({100*reachable_count/total_states:.1f}%)")

# Find shortest path
min_steps = np.min(controller.V[controller.V >= 0])
print(f"Shortest path: {min_steps} steps")

# Find states with no control
uncontrollable = np.sum(controller.h < 0)
print(f"Uncontrollable states: {uncontrollable}")
```

---

## Discretization Parameters

### Choosing Cell Counts

**Trade-off**: Finer discretization → Better approximation, Higher computation cost

| Problem | Typical Choices |
|---------|-----------------|
| 2D system | 5-10 cells/dim |
| 3D system | 4-6 cells/dim |
| 4D system | 3-4 cells/dim |
| >5D | 2-3 cells/dim (may need specialized techniques) |

### Angular Dimensions

For systems with angles (e.g., robot orientation), specify which dimensions are angular:

```python
# If dimension 2 is an angle θ ∈ [0, 2π]
angular_dims_x = [2]

controller = SymbolicController(
    ...
    angular_dims_x=angular_dims_x,
    ...
)
```

The framework automatically wraps reachable sets to [-π, π].

---

## Troubleshooting

### Common Issues

#### Issue: "No reachable states"
- **Cause**: Target region is unreachable or too conservative discretization
- **Solution**: 
  - Check target region is achievable with available controls
  - Verify bounds and discretization parameters
  - Reduce disturbance bounds for testing

#### Issue: "Controller returns None"
- **Cause**: Simulation reached unreachable state
- **Solution**:
  - Verify initial state is in reachable set
  - Check specification transition are complete
  - Increase `max_iter` in synthesis

#### Issue: "Slow computation"
- **Cause**: State space too large
- **Solution**:
  - Reduce cells per dimension
  - Lower system dimension if possible
  - Check for computation bottlenecks

#### Issue: "Control diverges"
- **Cause**: Disturbance larger than expected or specification inconsistent
- **Solution**:
  - Verify disturbance bounds are correct
  - Check Jacobian magnitudes
  - Validate specification transition

---

## Advanced Usage

### Custom Abstraction Refinement

For improved accuracy with larger state spaces:

```python
# Use a finer discretization only in critical regions
cells_per_dim = np.array([10, 5, 3])  # Different resolutions

# Multi-level approach:
# 1. Coarse synthesis
controller_coarse = SymbolicController(..., cells_per_dim_x=[3,3,3])
# 2. Refine around reachable set
# 3. Fine synthesis
controller_fine = SymbolicController(..., cells_per_dim_x=[10,5,3])
```

### Saving and Loading Results

```python
# After synthesis
import pickle

# Save
with open('controller_results.pkl', 'wb') as f:
    pickle.dump({'V': controller.V, 'h': controller.h}, f)

# Load
with open('controller_results.pkl', 'rb') as f:
    results = pickle.load(f)
    controller.V = results['V']
    controller.h = results['h']
```

---

## Performance Characteristics

### Computational Complexity

- **Symbolic abstraction**: O(N_x · N_u · T) where T is reachable set computation time
- **Reachability synthesis**: O(N_x · N_u · max_iter) with typical convergence in 10-100 iterations
- **Overall**: Polynomial in state/control discretization, manageable for ≤10⁵ states

### Memory Usage

- Transition relation T: ~8 · N_x · N_u bytes
- Value function V: ~4 · N_x bytes
- Control policy h: ~4 · N_x bytes

Example: 1000 states × 20 controls ≈ 160 KB

---

## References

For theoretical foundations, see the accompanying research paper.

### Key Concepts

- **Symbolic abstraction**: Discretize continuous system to finite model
- **Interval arithmetic**: Rigorously compute reachable sets with Jacobian bounds
- **Fixed-point iteration**: Compute largest/smallest controllable sets
- **Product automaton**: Combine system and specification dynamics

---

## Contributing

Contributions welcome! Please:
1. Follow the existing code style
2. Add docstrings to new functions
3. Update examples for new features
4. Test with multiple discretization levels

---

## License

[Specify your license here]

---

## Contact

For questions or issues, please open an issue on the repository or contact the authors.

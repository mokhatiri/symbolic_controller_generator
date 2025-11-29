# Symbolic Abstraction for Nonlinear Systems Control

## Quick Start (Simple Explanation)

**What is symbolic abstraction?**:<br>
It's a technique to convert a continuous system (smooth equations) into a discrete system (finite number of states and actions). This allows us to use formal methods and algorithms designed for discrete systems to synthesize guaranteed-correct controllers.

**Why is it useful?** 
- Guarantees controller correctness "by construction"
- Handles complex nonlinear systems
- Manages bounded disturbances and uncertainties
- No need to tune parameters or solve difficult optimization problems

**The three-step process:**
1. **Discretize**: Divide the continuous state/control space into a grid of cells
2. **Compute transitions**: For each cell pair (state, control), figure out which cells you can reach
3. **Synthesize controller**: Use discrete algorithms to find a controller that satisfies your specifications

## System Model

We work with continuous systems:

$ x^+ = f(x, u, w) $

**Meaning:**
- **x**: Current state (e.g., position, velocity) - **continuous values**
- **u**: Control input (e.g., motor force)
- **w**: Disturbance (e.g., wind, noise)
- **f**: How the system evolves
- $x^+$: Next state

**All values are bounded:**
- State stays in box: x in [x_min, x_max]
- Control limited: u in [u_min, u_max]
- Disturbance bounded: w in [w_min, w_max]

## Discretization (Making it Finite)

### Dividing the State Space into Grid Cells

Instead of working with infinite continuous values, we partition the state space into a **grid**:

```
Example: 3D System (x1, x2, x3)
State bounds: x1 in [0, 10], x2 in [0, 10], x3 in [0, 2*pi]
Grid cells: 100 cells in x1, 100 cells in x2, 30 cells in x3

Cell sizes:
- dx1 = 0.1 (since 10/100 = 0.1)
- dx2 = 0.1 (since 10/100 = 0.1)  
- dx3 = 0.21 radians (since 2*pi/30 ≈ 0.21)

Total cells = 100 x 100 x 30 = 300,000 states
```

**In general:**
- Cell width: dx (you choose this)
- Number of cells per dimension: K_i = ceil((x_i_max - x_i_min) / dx_i)
- Total number of cells: N_x = K_1 x K_2 x ... x K_n

### Numbering the Cells (Indexing)

With multiple dimensions, we need a way to **label each cell with a single number** (index).

**Simple approach**: Treat the grid like an array - go left to right, bottom to top.

```
Example: 2D grid with 3x2 cells
Cell coordinates (i,j) to Index state_idx

j=2: [(1,2)->4]  [(2,2)->5]  [(3,2)->6]
j=1: [(1,1)->1]  [(2,1)->2]  [(3,1)->3]
     i=1        i=2        i=3

Coordinates (2,1) to Index 2
Index 5 to Coordinates (2,2)
```

**Implementation**: Multiplier array M = [1, 3, 6]
- M[0] = 1 (always)
- M[1] = 3 (cells in first dimension)
- M[2] = 6 (total cells in first two dimensions)

This is **automatically computed** - you just specify [3, 2] (cells per dim) and the code determines the Index.

## Computing Transitions (The Core Algorithm)

**Goal:** For each pair (current state cell, control input), figure out which cells you can reach in the next time step.

### High-Level Idea

1. **Pick a state cell and control** (e.g., state cell #5, control input #2)
2. **Find the center point** of that state cell
3. **Simulate one step forward** using the dynamics: $x_{\text{new}} = f(x_{\text{old}}, u, w)$
4. **Account for uncertainty**:
   - The cell is not a point - you could start anywhere in the cell
   - Disturbances vary within bounds
   - Use Jacobians to estimate how far the result can deviate
5. **Find all cells** that the reachable set intersects
6. **Store the result** as a range: [min_cell, max_cell]

### Step-by-Step Example

**Setup:** 
- Current state cell: cell #3 (representing $x \in [1, 2]$)
- Control input: $u = 0.5$ N
- Disturbance center: $w = 0$ (no disturbance)

**Step 1: Cell center**
x_center = 1.5 m

**Step 2: Simulate dynamics**
dx = 2u - 0.5x = 2(0.5) - 0.5(1.5) = 0.85
x_next_center = 1.5 + 0.85 = 2.35 m

**Step 3: Add uncertainty**
- State cell uncertainty: +/- 0.5 m (cell width is 1m)
- Disturbance uncertainty: +/- 0.1 m (bounded disturbance)
- Jacobian effect: |df/dx| = 0.5, so uncertainty gets scaled

dx_next = 0.5 x 0.5 x 0.5 + 0.5 x 0.1 = 0.175 m

**Step 4: Reachable interval**
R = [2.35 - 0.175, 2.35 + 0.175] = [2.175, 2.525]

**Step 5: Map back to cells**
- Cell boundaries: ..., [2, 3), [3, 4), ...
- Cells covering [2.175, 2.525]: cell #3 and cell #4

**Result:** T[cell3, control2] = [3, 4]

### Angular Wraparound (Special Case)

If your system has angles (rotation, heading direction), they're periodic: 360 degrees = 0 degrees.

**The problem:** If reachable set is [350 degrees, 10 degrees], crossing the wraparound boundary 0 degrees:
- Don't store it as [350, 10] (nonsensical interval)
- Adjust: [350, 10] -> [350, 370] (mathematically cleaner)

The code automatically detects and fixes this.

### The Transition Matrix

All transitions are stored in a 3D matrix:
```
T[state_cell, control_input, 0] = minimum reachable cell
T[state_cell, control_input, 1] = maximum reachable cell
```

**Example:** 100 state cells x 20 control inputs = 100x20x2 array of integers.

This is the **finite symbolic model** - much simpler than the original continuous system!

## Code Implementation

#### Static Method: Build Multiplier Array

```python
def build_multiplier_array(cells_per_dim):
    """Automatically construct multiplier array from cells per dimension."""
```

Automatically converts cell counts to the efficient multiplier array representation. This eliminates manual computation.

#### Constructor Parameters (What You Need to Provide)

```python
abstraction = SymbolicAbstraction(
    f_dynamics, J_x, J_w,           # Your system's equations
    X_bounds, U_bounds,              # State and control limits
    dx_cell, cells_per_dim_x,        # How to discretize state space
    du_cell, cells_per_dim_u         # How to discretize control space
)
```


The constructor automatically computes:
- Multiplier arrays (for efficient indexing)
- Total number of states: N_x = 100 x 100 x 30 = 300,000
- Total number of controls: N_u = 3 x 5 = 15

#### Key Methods (What You Can Do)

| Method | Purpose |
|--------|---------|
| `abstraction.initialize()` | Compute all transitions (main computation) |
| `abstraction.save_symbolic_model(filename)` | Save result to disk for later |
| `abstraction.load_symbolic_model(filename)` | Reload previously saved model |
| `abstraction.T[state, control]` | Access transition for state/control pair |

#### Complete Usage Example

```python
import numpy as np
from AbstractSpace import SymbolicAbstraction

# Define your system: 3D nonlinear dynamics
def f(x, u, w):
    # Example: mobile robot or 2D position + angle system
    dx1 = u[0] * np.cos(x[2]) - 0.1*x[0] + w[0]
    dx2 = u[0] * np.sin(x[2]) - 0.1*x[1] + w[1]
    dx3 = u[1] - 0.05*x[2]  # angular velocity
    return np.array([dx1, dx2, dx3])

J_x = lambda u: np.array([[-0.1, 0, -u[0]*np.sin(x[2])],
                           [0, -0.1, u[0]*np.cos(x[2])],
                           [0, 0, -0.05]])  # Jacobian w.r.t. state
J_w = lambda u: np.array([[1, 0], [0, 1], [0, 0]])  # Jacobian w.r.t. disturbance

# Set up discretization from thesis tables
X_bounds = np.array([[0, 10], [0, 10], [0, 2*np.pi]])  # 3D state bounds
U_bounds = np.array([[-1, 1], [-1, 1]])                # 2D control bounds
dx_cell = np.array([0.1, 0.1, 0.21])                   # Cell sizes
cells_per_dim_x = np.array([100, 100, 30])             # 300,000 total states

du_cell = np.array([0.4, 0.4])      # Control cell size 0.4
cells_per_dim_u = np.array([3, 5])   # 15 total control inputs

# Create abstraction
abstraction = SymbolicAbstraction(
    f, J_x, J_w,
    X_bounds, U_bounds,
    dx_cell, cells_per_dim_x,
    du_cell, cells_per_dim_u
)

# Compute the transitions
abstraction.initialize()  # This takes time proportional to N_x x N_u (300,000 x 15)

# Now use the discrete model
# abstraction.T[state_idx, control_idx] = [min_next_state, max_next_state]
print(abstraction.T[0, 0])  # Where can we go from state 0 with control 0?
```

## Understanding the Key Formula

### The Taylor Expansion Formula (Implemented in Code)

The reachable set is computed using first-order Taylor expansion with interval arithmetic.

**Mathematical formulation:**

For an initial state cell cl(x_cell) = [x_center - dx_cell/2, x_center + dx_cell/2],
control u_control, and disturbance bounds W = [w_min, w_max]:

The reachable set R is bounded by:

R ⊆ [f(x_center, u_control, w_center) - D_x*dx - D_w*dw,
     f(x_center, u_control, w_center) + D_x*dx + D_w*dw]

Where:
- x_center = center of the state cell
- w_center = (w_min + w_max) / 2 (center of disturbance bounds)
- dx_succ = 0.5 * |J_x| * dx_cell + 0.5 * |J_w| * dw_bounds (total uncertainty)
- J_x = df/dx (Jacobian w.r.t. state)
- J_w = df/dw (Jacobian w.r.t. disturbance)
- |J_x|, |J_w| = absolute values (element-wise) for conservative interval arithmetic
- dw_bounds = w_max - w_min (width of disturbance bounds)

**How the code implements this:**

```python
# 1. Compute nominal successor from cell center
x_succ_center = f_dynamics(x_center, u_control, w_center)

# 2. Compute uncertainty bounds
dx_succ = 0.5 * abs(J_x(u_control)) @ dx_cell + 0.5 * abs(J_w(u_control)) @ dw_bounds

# 3. Construct interval bounds
R_min = x_succ_center - dx_succ  # Lower bound
R_max = x_succ_center + dx_succ  # Upper bound
R = [R_min, R_max]
```

This ensures all possible trajectories starting in the state cell, under any disturbance in bounds, are contained within the computed interval.


**Breaking down the uncertainty components:**

1. **f(x_center, u_control, w_center)**: Nominal trajectory
   - The function value evaluated at the cell center and disturbance center
   - Represents the expected/best-estimate next state

2. **0.5 * |J_x| * dx_cell**: State uncertainty propagation
   - dx_cell is the full width of the state cell
   - J_x (Jacobian df/dx) quantifies system sensitivity to state variations
   - The factor 0.5 accounts for the cell being centered (deviation is ±dx_cell/2 from center)
   - |J_x| * (0.5 * dx_cell) bounds how initial cell uncertainty spreads to the next state

3. **0.5 * |J_w| * dw_bounds**: Disturbance impact on reachability
   - dw_bounds = w_max - w_min is the full width of disturbance bounds
   - J_w (Jacobian df/dw) quantifies how disturbances affect the next state
   - The factor 0.5 accounts for disturbances being centered (deviation is ±dw_bounds/2 from center)
   - |J_w| * (0.5 * dw_bounds) bounds the worst-case disturbance effect

4. **Interval construction**: Conservative bounding
   - Lower bound: f(x_center, u_control, w_center) - (0.5*|J_x|*dx_cell + 0.5*|J_w|*dw_bounds)
   - Upper bound: f(x_center, u_control, w_center) + (0.5*|J_x|*dx_cell + 0.5*|J_w|*dw_bounds)
   - This creates a guaranteed enclosure containing all reachable states
   - The absolute value |·| ensures conservatism (includes both positive and negative deviations)

**Why this approach works:**

Using first-order Taylor expansion around the cell center with interval arithmetic provides a conservative, computationally efficient approximation of the exact nonlinear reachable set. All trajectories starting anywhere in the discretized cell and experiencing any bounded disturbance are guaranteed to stay within the computed interval bounds.

## Summary: How It All Works Together

```
+-----------------------------------------------------+
|  Continuous Nonlinear System                        |
|  dx = f(x, u, w)                                    |
|  (Infinite states, infinite controls)               |
+-----------------------------------------------------+
                      |
                      v
+-----------------------------------------------------+
|  1. DISCRETIZE: Grid cells                          |
|     State: 300,000 cells (100 x 100 x 30)           |
|     Control: 15 inputs (3 x 5)                      |
+-----------------------------------------------------+
                      |
                      v
+-----------------------------------------------------+
|  2. COMPUTE TRANSITIONS                             |
|     For each (cell, control), find reachable cells  |
|     -> 300,000 x 15 = 4,500,000 transitions         |
+-----------------------------------------------------+
                      |
                      v
+-----------------------------------------------------+
|  Finite Symbolic Model                              |
|  T[state, control] = [min_next, max_next]           |
|  (4,500,000 transitions instead of infinity)        |
+-----------------------------------------------------+
                      |
                      v
+-----------------------------------------------------+
|  3. SYNTHESIZE: Formal algorithms                   |
|     Find discrete controller satisfying specs       |
|     -> Guaranteed to work on the real system        |
+-----------------------------------------------------+
```

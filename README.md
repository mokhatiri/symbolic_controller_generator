# Symbolic Controller Generator

A Python toolkit for **symbolic (discrete) controller synthesis** of continuous dynamical systems. It discretizes a continuous state space, builds a finite-state transition map based on attainability analysis, composes it with a specification automaton, and extracts a provably correct controller through reachability and safety pruning.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [API Reference](#api-reference)
- [Running the Example](#running-the-example)

---

## Overview

Symbolic control theory bridges the gap between continuous control systems and formal verification. This project implements the core pipeline:

1. **Discretize** a 3D continuous state space into a finite grid of cells.
2. **Compute attainability**: for each cell and each control input, determine which cells the system can reach next (accounting for disturbances and model uncertainty).
3. **Build a transition map** (`TransMap`) — a finite automaton over the discrete state space.
4. **Compose** the transition map with a specification automaton to form a **product automaton**.
5. **Prune** the product automaton using fixed-point reachability (reach a target) and safety (stay within safe states) algorithms.
6. **Extract a controller** `h: state → control` that guarantees the specification is met.

---

## How It Works

### State Space Discretization

The continuous space is parameterized as:

```
space = (x1_max, x2_max, x3_range, NX1, NX2, NX3)
```

This defines a 3D grid with `NX1 × NX2 × NX3` cells. Each cell `(i, j, k)` maps back to a continuous bounding box via `StateC`.

### Attainability (`Attein`)

For each cell and control `u`, the next reachable set is over-approximated as:

```
x_next = F(x_center, u, w) ± (Dx · Sx + Dw · Sw)
```

where `Dx`, `Dw` are sensitivity matrices and `Sx`, `Sw` quantify state and disturbance uncertainty. All cells intersecting this bounding box become successors in the transition map.

### Product Automaton

A specification is given as a finite automaton `SpecMap: state × label → next_state`. A `LabelMap` assigns a label to each discrete cell. The product automaton pairs system states with specification states `(x, q)`, and transitions follow both systems simultaneously.

### Pruning

- **Reachability** (`ApplyReachability`): backward fixed-point computation — keep only states from which the target set is reachable.
- **Safety** (`ApplySecurity`): keep only states where every successor stays within the safe set (for all disturbances).

---

## Project Structure

```
symbolic_controller_generator/
├── utils.py           # Space discretization, attainability, TransMap construction, label utilities
├── Automata.py        # Automata class: product automaton, reachability/safety pruning, BFS, CSV I/O
├── run_pipeline.py    # Minimal smoke-test / pipeline demo script
├── Example.ipynb      # Interactive notebook with a full worked example and visualizations
├── requirements.txt   # Python dependencies
└── README.md
```

### `utils.py`

| Function | Description |
|---|---|
| `TransMap(...)` | Builds the full discrete transition map using attainability |
| `Attein(...)` | Computes the reachable bounding box for one cell + control |
| `SpaceD(...)` | Returns all cells inside a continuous bounding box |
| `SpaceBound(...)` | Returns the min/max cell indices for a bounding box |
| `StateC(state, space)` | Converts a discrete cell index to its continuous bounds |
| `ControlD(controls)` | Enumerates the discrete control set |
| `StateInArea(...)` | Tests whether a cell lies within a region |
| `build_LabelMap(...)` | Assigns labels to cells based on region definitions |
| `cell_center(...)` | Returns the continuous center coordinates of a cell |

### `Automata.py`

| Method | Description |
|---|---|
| `__init__(TransMap)` | Initializes with a transition map |
| `ToProdAutomate(SpecMap, LabelMap, start_spec)` | Builds the product automaton via BFS |
| `ApplyReachability(target_states)` | Backward reachability fixed-point pruning |
| `ApplySecurity(correct_states)` | Safety fixed-point pruning |
| `bfs_trajectory(startStates, goalStates)` | Finds a trajectory using the extracted controller `h` |
| `getH()` | Returns the synthesized controller mapping `state → control` |
| `SaveAutomataToCsv(path)` | Serializes the transition map to CSV |
| `LoadAutomataFromCsv(path)` | Loads a transition map from CSV |

---

## Installation

**Requirements:** Python 3.8+

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `matplotlib`.

---

## Quickstart

```python
import numpy as np
from utils import TransMap, build_LabelMap
from Automata import Automata

# 1. Define the state space and control grid
space    = (1.0, 1.0, (-1, 1), 10, 10, 10)   # x1_max, x2_max, x3_range, NX1, NX2, NX3
controls = (0.0, 1.0, 0.0, 1.0, 3, 3)         # u1_min, u1_max, u2_min, u2_max, NU1, NU2
inputs   = ([0.0, 0.0, 0.0], 0.1)             # disturbance bounds, time step

x0_min = (0.0, 0.0, -1.0)
x0_max = (1.0, 1.0,  1.0)

# 2. Define system dynamics and uncertainty (implement these for your system)
def TransFunct(x_center, u, w, tau): ...
def Dx(u, tau):   ...   # sensitivity matrix w.r.t. state uncertainty
def Dw(tau):      ...   # sensitivity matrix w.r.t. disturbance
def Sx(xmin, xmax): ... # state uncertainty bound
def Sw(w):        ...   # disturbance bound
def Xx(xmin, xmax): ... # representative state point (e.g., center)
def Wx():         ...   # representative disturbance point

# 3. Build the transition map
Tmap = TransMap(x0_min, x0_max, controls, inputs, space,
                TransFunct, Dx, Dw, Sx, Sw, Xx, Wx)

# 4. Build label map and specification automaton
Rs = {
    1: (((0.8, 0.8, -0.2), (1.0, 1.0, 0.2)), True),  # target region → label 1
}
LabelMap, _ = build_LabelMap(space, x0_min, x0_max, Rs, default=0)

SpecMap = {
    'init': {0: 'init', 1: 'goal'},
    'goal': {0: 'goal', 1: 'goal'},
}

# 5. Build product automaton and synthesize controller
A = Automata(Tmap)
A.ToProdAutomate(SpecMap, LabelMap, 'init')

target_states = [s for s in A.TransMap if s[1] == 'goal']
A.ApplyReachability(target_states)

# 6. Extract and use the controller
controller = A.getH()   # dict: product_state → control_input
```

---

## Running the Example

**Notebook (recommended):** Open `Example.ipynb` in JupyterLab or VS Code for a full worked example with step-by-step explanations and trajectory visualizations.

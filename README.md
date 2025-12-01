# Symbolic Controller Generator

This small Python project builds a symbolic (discrete) abstraction of a
continuous dynamical system, constructs product automata with a
specification machine, prunes the automaton using reachability and safety
conditions, and provides simple visualization utilities.

Key components:

- `utils.py`: discretization helpers, atteinability-based transition map
	construction (`TransMap`), region/label building, and geometric helpers.
- `Automata.py`: `Automata` class that stores a `TransMap` and exposes
	methods to build the product automaton (`ToProdAutomate`), prune using
	reachability (`ApplyReachability`) and safety (`ApplySecurity`), BFS
	trajectory search, and CSV load/save helpers.
- `Example.ipynb`: an interactive notebook demonstrating how to define the
	continuous dynamics, generate a `TransMap`, build the product automaton,
	and visualize trajectories.

Quickstart
----------

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. From Python, generate a `TransMap` and run the basic pipeline:

```python
from utils import TransMap
from Automata import Automata

# define space, controls, inputs and helper functions (see Example.ipynb)
# Tmap = TransMap(x0_min, x0_max, controls, inputs, space, TransFunct, Dx, Dw, Sx, Sw, Xx, Wx)
A = Automata(Tmap)
# Build product automaton: A.ToProdAutomate(SpecMap, LabelMap, start_spec)
# Prune with reachability: A.ApplyReachability(target_states)
```

Files
-----

- `utils.py` — discretization and transition map builder.
- `Automata.py` — automaton class and utilities.
- `Example.ipynb` — example workflow and visualizations.
- `requirements.txt` — Python package dependencies.    
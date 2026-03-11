"""
Smoke-test pipeline
===================
Generates a minimal TransMap, builds the product automaton, applies
reachability pruning, and prints a sample of the resulting transitions.
Intended as a quick sanity-check that the full pipeline runs end-to-end.
"""

from pprint import pprint

import numpy as np

from Automata import Automata
from utils import TransMap

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# State space: (x1_max, x2_max, x3_range, NX1, NX2, NX3)
SPACE = (1.0, 1.0, (-1, 1), 2, 2, 2)

# Control grid: (u1_min, u1_max, u2_min, u2_max, NU1, NU2)
CONTROLS = (0.0, 0.0, 0.0, 0.0, 1, 1)  # single control point

# Inputs / disturbances
INPUTS = (0.0, 0.0, 0.0, 1.0)

# Bounding box of the initial state space
X0_MIN = (0.0, 0.0, SPACE[2][0])
X0_MAX = (SPACE[0], SPACE[1], SPACE[2][1])

# Specification automaton: single state 'a' that loops on every label
SPEC_MAP = {"a": {0: "a"}}
START_SPEC = "a"

# Number of sample transitions to display after pruning
N_SAMPLE = 5

# ---------------------------------------------------------------------------
# System dynamics  (identity — center stays put, zero uncertainty)
# ---------------------------------------------------------------------------

def TransFunct(Xx, u, inputs):
    """Trivial identity dynamics: next center equals current center."""
    return tuple(Xx)


def Dx(u, inputs):
    """State-uncertainty sensitivity matrix (zero for this test)."""
    return np.zeros((3, 3), dtype=float)


def Dw(inputs):
    """Disturbance-uncertainty sensitivity matrix (zero for this test)."""
    return np.zeros((3, 3), dtype=float)


def Sx(x_min, x_max, inputs):
    """State uncertainty bound (zero for this test)."""
    return np.zeros(3, dtype=float)


def Sw(inputs):
    """Disturbance uncertainty bound (zero for this test)."""
    return np.zeros(3, dtype=float)


def Xx(x_min, x_max, inputs):
    """Representative state point: cell centre."""
    return np.array([(x_min[i] + x_max[i]) / 2 for i in range(3)], dtype=float)


def Wx(inputs):
    """Representative disturbance point: zero."""
    return [0, 0, 0]

# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def build_transition_map():
    print("Step 1 — Building transition map...")
    Tmap = TransMap(
        X0_MIN, X0_MAX, CONTROLS, INPUTS, SPACE,
        TransFunct, Dx, Dw, Sx, Sw, Xx, Wx,
    )
    print(f"          {len(Tmap)} discrete states generated.\n")
    return Tmap


def build_product_automaton(Tmap):
    print("Step 2 — Building product automaton...")
    LabelMap = {s: 0 for s in Tmap}
    A = Automata(Tmap)
    A.ToProdAutomate(SPEC_MAP, LabelMap, START_SPEC)
    print(f"          {len(A.TransMap)} product states.\n")
    return A


def apply_reachability(A):
    print("Step 3 — Applying reachability pruning...")
    targets = [p for p in A.TransMap if p[1] == START_SPEC]
    print(f"          {len(targets)} target states.")
    A.ApplyReachability(targets)
    print(f"          {len(A.TransMap)} states remaining after pruning.\n")


def print_sample(A):
    print(f"Sample transitions (up to {N_SAMPLE}):")
    for i, (state, ctrl) in enumerate(A.TransMap.items()):
        if i >= N_SAMPLE:
            break
        pprint((state, ctrl))

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    Tmap = build_transition_map()
    A = build_product_automaton(Tmap)
    apply_reachability(A)
    print_sample(A)
    print("Done.")


if __name__ == "__main__":
    main()


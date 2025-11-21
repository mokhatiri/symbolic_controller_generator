"""Quick pipeline test: generate a tiny TransMap, build product automaton,
and run reachability pruning. Intended as a minimal smoke test.
"""
from pprint import pprint
from utils import TransMap
from Automata import Automata
import numpy as np


def TransFunct(Xx, u, inputs):
    # simple identity dynamics (center stays the same)
    return tuple(Xx)


def Dx(u, inputs):
    return np.zeros((3, 3), dtype=float)


def Dw(inputs):
    return np.zeros((3, 3), dtype=float)


def Sx(x_min, x_max, inputs):
    return np.zeros(3, dtype=float)


def Sw(inputs):
    return np.zeros(3, dtype=float)


def Xx(x_min, x_max, inputs):
    return np.array([(x_min[i] + x_max[i]) / 2 for i in range(3)], dtype=float)


def Wx(inputs):
    return [0, 0, 0]


def main():
    # tiny discretization to keep transitions small
    space = (1.0, 1.0, (-1, 1), 2, 2, 2)  # NX1=2,NX2=2,NX3=2
    controls = (0.0, 0.0, 0.0, 0.0, 1, 1)  # single control
    inputs = (0.0, 0.0, 0.0, 1.0)

    x0_min = (0.0, 0.0, space[2][0])
    x0_max = (space[0], space[1], space[2][1])

    print("Generating TransMap...")
    Tmap = TransMap(x0_min, x0_max, controls, inputs, space,
                    TransFunct, Dx, Dw, Sx, Sw, Xx, Wx)

    print(f"Generated TransMap with {len(Tmap)} states")

    A = Automata(Tmap)

    # simple LabelMap and SpecMap: single label 0, one spec state 'a' looping
    LabelMap = {s: 0 for s in Tmap.keys()}
    SpecMap = {'a': {0: 'a'}}

    print("Building product automaton...")
    A.ToProdAutomate(SpecMap, LabelMap, 'a')
    print(f"Product automaton states: {len(A.TransMap)}")

    # targets: all product states whose spec component is 'a'
    targets = [p for p in A.TransMap.keys() if p[1] == 'a']
    print(f"Target product states: {len(targets)}")

    print("Applying reachability pruning...")
    A.ApplyReachability(targets)
    print(f"Pruned product automaton states: {len(A.TransMap)}")

    # show a few entries
    print("Sample transitions after pruning:")
    for i, (s, ctrl) in enumerate(A.TransMap.items()):
        if i >= 5:
            break
        pprint((s, ctrl))

    print("Done.")


if __name__ == '__main__':
    main()

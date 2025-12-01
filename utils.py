import math
import numpy as np

def SpaceBound(x_min, x_max, space, intersect = False):
    """
    takes a Space defined by an area : x_min, x_max
    and returns the max and min cell indices inside the area (if intersect == true: the cells intersecting the area);

    space = (x1_max, x2_max, x3_range, NX1, NX2, NX3) // x3 are multipliers of pi

    input: x_min, x_max, space, intersect
    output: S_min, S_max
    """
    (x1_max, x2_max, x3_range, NX1, NX2, NX3) = space
    xi=[(0,x1_max),(0,x2_max),x3_range]
    Nx=[NX1,NX2,NX3]
    # start by getting the cell at x_min and x_max
    S_max=[]
    S_min=[]
    border_max=[]
    border_min=[]
    for i in range(3):
        L = xi[i][1] - xi[i][0]
        coords_max = Nx[i] * (x_max[i] - xi[i][0]) / L
        coords_min = Nx[i] * (x_min[i] - xi[i][0]) / L

        if intersect:
            idx_max = math.floor(coords_max)
            if coords_max.is_integer():
                idx_max -= 1
            idx_min = math.floor(coords_min)
        else:
            idx_max = math.floor(coords_max) - 1
            idx_min = math.floor(coords_min)
            if not coords_min.is_integer():
                idx_min += 1

        # clamp to valid range [0, Nx[i] - 1]
        idx_max = max(0, min(Nx[i] - 1, idx_max))
        idx_min = max(0, min(Nx[i] - 1, idx_min))

        S_max.append(idx_max)
        S_min.append(idx_min)

    return S_min, S_max

def SpaceD(x_min, x_max, space, intersect = False):
    """
    takes a Space defined by an area : x_min, x_max
    and returns the cells inside the area (if intersect == true: the cells intersecting the area);

    space = (x1_max, x2_max, x3_range, NX1, NX2, NX3) // x3 are multipliers of pi

    input: x_min, x_max, space, intersect
    output: cells
    """

    S_min, S_max = SpaceBound(x_min, x_max, space, intersect)

    cells=[]
    for i in range(max(S_min[0], 0),min(S_max[0]+1, space[3])):
        for j in range(max(S_min[1], 0),min(S_max[1]+1, space[4])):
            for k in range(max(S_min[2], 0),min(S_max[2]+1, space[5])):
                cells.append((i,j,k))
    return cells

def StateInArea(state, x_min, x_max, space, intersect=False):
    """
    Returns True if the cell `state` intersects or is fully included in the area [x_min, x_max]
    """
    cell_min, cell_max = StateC(state, space)
    
    if intersect:
        # check if the cell and area intersect
        overlap = all(cell_max[d] >= x_min[d] and cell_min[d] <= x_max[d] for d in range(3))
        return overlap
    else:
        # check if the cell is fully inside the area
        inside = all(cell_min[d] >= x_min[d] and cell_max[d] <= x_max[d] for d in range(3))
        return inside

def ControlD(controls):
    """
    returns the discrete control set defined by controls
    """
    (u1_min, u1_max, u2_min, u2_max, NU1, NU2) = controls
    u1_vals = np.linspace(u1_min, u1_max, NU1)
    u2_vals = np.linspace(u2_min, u2_max, NU2)
    return [(u1, u2) for u1 in u1_vals for u2 in u2_vals]

def StateC(state, space):
    """
        takes a state (cell coordinates) and a space definition
        and returns the continuous x_min and x_max for that cell.

        state = (i, j, k)
        space = (x1_max, x2_max, x3_range, NX1, NX2, NX3) // x3 are multipliers of pi

        input: state, space
        output: x_min_coords, x_max_coords (tuples of continuous values)
    """
    (x1_max, x2_max, x3_range, NX1, NX2, NX3) = space
    xi=[(0,x1_max),(0,x2_max),x3_range]
    Nx=[NX1,NX2,NX3]

    x_min_coords = [0.0] * 3
    x_max_coords = [0.0] * 3

    for dim in range(3):
        idx = state[dim]
        L = xi[dim][1] - xi[dim][0]

        # Calculate x_min for the current dimension
        x_min_coords[dim] = (idx / Nx[dim]) * L + xi[dim][0]

        # Calculate x_max for the current dimension
        x_max_coords[dim] = ((idx + 1) / Nx[dim]) * L + xi[dim][0]

    return tuple(x_min_coords), tuple(x_max_coords)

def Attein(x_min, x_max, u, inputs, TransFunct, Dx_, Dw_, Sx_, Sw_, Xx_, Wx_):
    """
    x_max, x_min : listes/tuples de taille 3
    u           : tuple/liste (u1, u2)
    inputs      : ((w1, w2, w3), tau)
    TransFunct  : fonction de transition
    Dx, Dw      : matrices numpy
    Sx, Sw      : numpy listes/tuples de taille 3
    Xx          : numpy liste/tuple de taille 3
    retourne (x_max_next, x_min_next)
    """

    Dx = Dx_(u, inputs[1])
    Dw = Dw_(inputs[1])
    Sx = Sx_(x_min, x_max)
    Sw = Sw_(inputs[0])
    Xx = Xx_(x_min, x_max)
    Wx = Wx_()

    valF = np.array(TransFunct(Xx, u, Wx, inputs[1]), dtype=float)
    delta = Dx @ Sx + Dw @ Sw

    x_min_next = valF - delta
    x_max_next = valF + delta

    return tuple(x_min_next), tuple(x_max_next)

def continuous_to_cell(pt, space):
    # pt = (x1, x2, x3_cont)
    x1_max, x2_max, x3_range, NX1, NX2, NX3 = space
    dx1 = x1_max / NX1
    dx2 = x2_max / NX2
    # x3 is in range [x3_range[0], x3_range[1]] scaled to NX3 cells
    x3_min = x3_range[0]
    x3_max = x3_range[1]
    dx3 = (x3_max - x3_min) / NX3

    i = int(min(NX1-1, max(0, math.floor(pt[0] / dx1))))
    j = int(min(NX2-1, max(0, math.floor(pt[1] / dx2))))
    k = int(min(NX3-1, max(0, math.floor((pt[2] - x3_min) / dx3))))
    return (i, j, k)


def cell_center(state, space):
    """
    Compute the continuous coordinates of the center of a discrete cell.
    
    state: tuple (i, j, k)
    space: (x1_max, x2_max, x3_range, NX1, NX2, NX3)
    
    Returns: tuple (x1_center, x2_center, x3_center)
    """
    i, j, k = state
    x1_max, x2_max, x3_range, NX1, NX2, NX3 = space
    
    # cell sizes
    dx1 = x1_max / NX1
    dx2 = x2_max / NX2
    dx3 = (x3_range[1] - x3_range[0]) / NX3

    # compute center
    x1_center = (i + 0.5) * dx1
    x2_center = (j + 0.5) * dx2
    x3_center = x3_range[0] + (k + 0.5) * dx3

    return (x1_center, x2_center, x3_center)

def TransMap(x0_min, x0_max, controls, inputs, space, TransFunct, Dx, Dw, Sx, Sw, Xx, Wx):
    """
    Compute the automata, defined using the atteinability of TransFunction, Dw, Dx, Sw, Sx, and Xx
    """
    SpcD=SpaceD(x0_min,x0_max,space)
    CntD=ControlD(controls)
    Tmap={}
    for state in SpcD:
        Statemap={}
        for u in CntD:
            x_min,x_max=StateC(state,space)
            x_min,x_max=Attein(x_min, x_max, u, inputs, TransFunct, Dx, Dw, Sx, Sw, Xx, Wx)
            Statemap[u]=SpaceD(x_min,x_max,space,True)
        Tmap[state] = Statemap

    return Tmap

def build_LabelMap(space, x0_min, x0_max, Rs, default = 0):
    all_states = SpaceD(x0_min, x0_max, space)

    region_list = [(label, rmin, rmax, intersect)
                   for label, ((rmin, rmax), intersect) in Rs.items()]

    label_map = {}
    default_states = []

    for state in all_states:
        assigned = default
        for label, rmin, rmax, intersect in region_list:
            if StateInArea(state, rmin, rmax, space, intersect=True):
                assigned = label
                break

        label_map[state] = assigned
        if(assigned == default):
            default_states.append(state)

    return label_map, default_states
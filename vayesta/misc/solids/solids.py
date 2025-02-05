import numpy as np

SQRT2 = np.sqrt(2)

def bcc(atoms, a):
    """Body-centered cubic.
    Li: 3.51 A
    """
    amat = a*np.eye(a)
    coords = a * np.asarray([[0, 0, 0], [1, 1, 1]])/2
    atom = _make_atom(atoms, coords)
    return amat, atom

def diamond(atoms=['C', 'C'], a=3.57):
    """Silicon: a=5.431 A"""
    amat = a * np.asarray([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5]])
    coords = a * np.asarray([[0, 0, 0], [1, 1, 1]])/4
    atom = _make_atom(atoms, coords)
    return amat, atom

def graphene(atoms=['C', 'C'], a=2.46, c=20.0):
    """
    hBN: a=2.5A
    hBeO: a=2.68A (VASP)"""
    amat = np.asarray([
            [a, 0, 0],
            [a/2, a*np.sqrt(3.0)/2, 0],
            [0, 0, c]])
    coords_internal = np.asarray([
        [2.0, 2.0, 3.0],
        [4.0, 4.0, 3.0]])/6
    coords = np.dot(coords_internal, amat)
    atom = _make_atom(atoms, coords)
    return amat, atom

def graphite(atoms=['C', 'C', 'C', 'C'], a=2.461, c=6.708):
    """a = 2.461 A , c = 6.708 A"""
    amat = np.asarray([
            [a/2, -a*np.sqrt(3.0)/2, 0],
            [a/2, +a*np.sqrt(3.0)/2, 0],
            [0, 0, c]])
    coords_internal = np.asarray([
        [0,     0,      1.0/4],
        [2.0/3, 1.0/3,  1.0/4],
        [0,     0,      3.0/4],
        [1.0/3, 2.0/3,  3.0/4]])
    coords = np.dot(coords_internal, amat)
    atom = _make_atom(atoms, coords)
    return amat, atom

def rocksalt(atoms=['Na', 'Cl'], a=5.6402, unitcell='primitive'):
    """
    LiH: a=4.0834
    LiF: a=4.0351
    """
    if unitcell == 'primitive':
        amat = a * np.asarray([
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5]])
        internal = np.asarray([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5]])
        coords = np.dot(internal, amat)
        atom = _make_atom(atoms, coords)
        return amat, atom

    if unitcell == 'primitive-af1':
        # wrong:
        #amat = a * np.asarray([
        #    [1/SQRT2,   0,          0],
        #    [0,         1/SQRT2,    0],
        #    [0,         0,          1]])
        #internal = np.asarray([
        #    [0.0, 0.0, 0.0],
        #    [0.5, 0.5, 0.0],
        #    [0.0, 0.0, 0.5],
        #    [0.5, 0.5, 0.5]])
        # arXiv:2207.12064v1:
        amat = a * np.asarray([
            [0.5,   0,  0.5],
            [0.5,   0,  -0.5],
            [0,     1,  0.0]])
        coords = a/2 * np.asarray([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0]])
        atom = _make_atom(2*atoms, coords)
        return amat, atom

    if unitcell == 'primitive-af2':
        amat = a * np.asarray([
            [1.00, 0.50, 0.50],
            [0.50, 1.00, 0.50],
            [0.50, 0.50, 1.00]])
        internal = np.asarray([
            [0.00, 0.00, 0.00],
            [0.25, 0.25, 0.25],
            [0.50, 0.50, 0.50],
            [0.75, 0.75, 0.75]])
        coords = np.dot(internal, amat)
        atom = _make_atom(2*atoms, coords)
        return amat, atom

    if unitcell == 'cubic':
        amat = a*np.eye(3)
        internal = np.asarray([
            # Atom 1:
            [0, 0, 0],
            [0, 1/2, 1/2],
            [1/2, 0, 1/2],
            [1/2, 1/2, 0],
            # Atom 2:
            [0, 0, 1/2],
            [0, 1/2, 0],
            [1/2, 0, 0],
            [1/2, 1/2, 1/2]])
        coords = np.dot(internal, amat)
        atom = _make_atom(4*[atoms[0]]+4*[atoms[1]], coords)
        return amat, atom
    raise ValueError

def perovskite(atoms=['Sr', 'Ti', 'O'], a=3.905):
    if len(atoms) == 3:
        atoms = [atoms[0], atoms[1]] + 3*[atoms[2]]
    amat = a * np.eye(3)
    coords = a*np.asarray([
                [0,     0,      0],
                [1/2,   1/2,    1/2],
                [0,     1/2,    1/2],
                [1/2,   0,      1/2],
                [1/2,   1/2,    0]
                ])
    atom = _make_atom(atoms, coords)
    return amat, atom

def perovskite_tetragonal(atoms=['Sr', 'Ti', 'O'], a=5.507, c=7.796, u=0.241):
    """This is the crystallographic ('quadruple') cell, not the primitive ('double') cell.

    Lattice constants from PHYSICAL REVIEW MATERIALS 2, 013807 (2018).
    DOI:10.1103/PhysRevMaterials.2.013807

    becomes cubic with parameters:
    a = 5.522 A, c = 2a0 = 7.810 A , u = 0.25
    (see DOI: 10.1103/PhysRevB.83.134108)
    """
    if a is None:
        a = c/np.sqrt(2)
    if c is None:
        c = np.sqrt(2)*a

    if len(atoms) == 3:
        atoms = [atoms[0], atoms[1]] + 3*[atoms[2]]
    if len(atoms) == 5:
        atoms = 4*atoms

    amat = a * np.eye(3)
    amat[2,2] = c

    #coords_internal = a*np.asarray([
    #            [0      ,   0.5     ,   0.25],  # Sr
    #            [0.5    ,   0       ,   0.75],  # Sr
    #            [0.5    ,   0       ,   0.25],  # Sr
    #            [0      ,   0.5     ,   0.75],  # Sr
    #            [0      ,   0       ,   0],     # Ti
    #            [0.5    ,   0.5     ,   0.5],   # Ti
    #            [0      ,   0       ,   0.5],   # Ti
    #            [0.5    ,   0.5     ,   0],     # Ti
    #            [0      ,   0       ,   0.25],  # O (4a)
    #            [0.5    ,   0.5     ,   0.75],  # O (4a)
    #            [0      ,   0       ,   0.75],  # O (4a)
    #            [0.5    ,   0.5     ,   0.25],  # O (4a)
    #            [u      ,   0.5+u   ,   0],     # O
    #            [0.5+u  ,   u       ,   0.5],   # O
    #            [-u     ,   0.5-u   ,   0],     # O
    #            [0.5-u  ,   -u      ,   0.5],   # O
    #            [0.5-u  ,   u       ,   0],     # O
    #            [-u     ,   0.5+u   ,   0.5],   # O
    #            [0.5+u  ,   -u      ,   0],     # O
    #            [u      ,   0.5-u   ,   0.5],   # O
    #            ])
    # Order?
    coords_internal = np.asarray([
                [0      ,   0.5     ,   0.25],  # Sr
                [0      ,   0       ,   0],     # Ti
                [0      ,   0       ,   0.25],  # O (4a)
                [u      ,   0.5+u   ,   0],     # O
                [0.5+u  ,   u       ,   0.5],   # O
                [0.5    ,   0       ,   0.75],  # Sr
                [0.5    ,   0.5     ,   0.5],   # Ti
                [0.5    ,   0.5     ,   0.75],  # O (4a)
                [-u     ,   0.5-u   ,   0],     # O
                [0.5-u  ,   -u      ,   0.5],   # O
                [0.5    ,   0       ,   0.25],  # Sr
                [0      ,   0       ,   0.5],   # Ti
                [0      ,   0       ,   0.75],  # O (4a)
                [0.5-u  ,   u       ,   0],     # O
                [-u     ,   0.5+u   ,   0.5],   # O
                [0      ,   0.5     ,   0.75],  # Sr
                [0.5    ,   0.5     ,   0],     # Ti
                [0.5    ,   0.5     ,   0.25],  # O (4a)
                [0.5+u  ,   -u      ,   0],     # O
                [u      ,   0.5-u   ,   0.5],   # O
                ])
    coords = np.dot(coords_internal, amat)

    atom = _make_atom(atoms, coords)
    return amat, atom

def _make_atom(atoms, coords):
    atom = []
    for i in range(len(atoms)):
        if atoms[i] is not None:
            atom.append([atoms[i], coords[i]])
    return atom

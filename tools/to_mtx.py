# -*- coding: utf-8 -*-
"""
to_mtx.py - write a case in Matrix Market format, for solvers outside Python (and the SuiteSparse conventions).

    python tools/to_mtx.py makro10            -> data/makro10/mtx/A.mtx, b_00.mtx, ..., x_00.mtx, ...

A.mtx is a 'coordinate real general' matrix, the vectors are 'array real general' n x 1. The text files are
several times larger than the npz/npy files (makro100: about 0.4 GB for A).
"""
import sys
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sps

ROOT = Path(__file__).resolve().parent.parent


def main(names):
    for name in names:
        folder = ROOT / 'data' / name
        out = folder / 'mtx'
        out.mkdir(exist_ok=True)
        A = sps.load_npz(folder / 'A.npz').tocoo()
        scipy.io.mmwrite(out / 'A.mtx', A, comment=f'factorbench case {name}: stacked Newton Jacobian J',
                         field='real', precision=17)
        for p in sorted(folder.glob('[bx]_*.npy')):
            scipy.io.mmwrite(out / f'{p.stem}.mtx', np.load(p).reshape(-1, 1), precision=17)
        print(f'{name}: -> {out}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])

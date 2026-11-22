# -*- coding: utf-8 -*-
"""
cases.py - the test cases: a sparse matrix J and the right hand sides b of one Newton solve.

A case is a folder data/<case>/ with
    A.npz        J as a scipy sparse matrix (CSR)
    b_00.npy ... the right hand sides, in the order Newton solved them
    x_00.npy ... the reference solutions (normal equations + PARDISO, refined to a relative residual of about 1e-12)
    meta.json    sizes, how the case was made, the reference timings
    names.json   (optional) the equation/unknown names of one period and the periods

get_data.py downloads the cases.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sps

DATA = Path(__file__).resolve().parent.parent / 'data'


@dataclass
class Case:
    name: str
    A: sps.csr_matrix
    bs: list
    xs: list
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return self.A.shape[0]

    def period_of(self):
        """The period number (0, 1, ...) of every row and column (the matrices are period-major)."""
        return np.arange(self.n) // self.meta['n_per_period']

    def describe(self):
        m = self.meta
        return (f'{self.name}: n={self.n:,}, nnz={self.A.nnz:,}, {len(self.bs)} right hand sides, '
                f'{m.get("zero_diagonals", "?"):,} zero diagonals, periods {m.get("periods")}')


def available(data=DATA):
    """The cases in the data folder."""
    data = Path(data)
    return sorted(p.name for p in data.iterdir() if (p / 'A.npz').exists()) if data.exists() else []


def load(name, data=DATA, rhs=None):
    """Load a case. rhs: a list of right hand side numbers (default all)."""
    folder = Path(data) / name
    if not (folder / 'A.npz').exists():
        raise FileNotFoundError(f'case {name!r} is not in {Path(data)}; run  python get_data.py {name}  '
                                f'(available here: {", ".join(available(data)) or "none"})')
    A = sps.load_npz(folder / 'A.npz').tocsr()
    meta = json.loads((folder / 'meta.json').read_text(encoding='utf-8'))
    numbers = sorted(int(p.stem[2:]) for p in folder.glob('b_*.npy'))
    if rhs is not None:
        missing = set(rhs) - set(numbers)
        if missing:
            raise ValueError(f'case {name}: no right hand side {sorted(missing)} (has {numbers})')
        numbers = list(rhs)
    bs = [np.load(folder / f'b_{k:02d}.npy') for k in numbers]
    xs = [np.load(folder / f'x_{k:02d}.npy') if (folder / f'x_{k:02d}.npy').exists() else None for k in numbers]
    meta['rhs_numbers'] = numbers
    return Case(name, A, bs, xs, meta)

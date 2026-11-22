# -*- coding: utf-8 -*-
"""
export_makro.py - write the linear systems of MAKRO's stacked Newton solve as test cases for factorbench.

Needs ModelFlow and the MAKRO 2026 translation (C:\\mfmodeller_raw\\makro\\makro26). It is NOT needed to run the
benchmark: it documents how the published data was made.

The ModelFlow option ``factorizer=`` takes a function ``factorizer(J) -> solve(b)``. The recorder below is such a
function: it saves J when ModelFlow factorizes it and every b (and the x it returns) when Newton solves with it,
and hands the actual work to the method MAKRO uses now (normal equations, PARDISO, refined against J).

The run is the one of makro26_experiment_simple.ipynb: a baseline solve (one factorization at the baseline), then
public purchases (QR__OFF) +1 % for the whole span, which Newton solves in a handful of iterations with the same
factorization. That is the pattern that matters: factorize once, solve many times.

    python export_makro.py makro100            # 2030-2129, the whole horizon (about 2 minutes, 20 GB)

Output: ../data/<case>/ with A.npz (scipy CSR), b_00.npy, b_01.npy ..., x_00.npy ... and meta.json.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sps

MAKRO = Path(r'C:\mfmodeller_raw\makro\makro26')
OUT = Path(__file__).resolve().parent.parent / 'data'
CASES = {'makro100': (2030, 2129)}
# the reference solve: as in makro26 (dream_shocks.FACTORIZER) but refined to the default 1e-11, so the saved x
# are accurate reference solutions
REFERENCE = {'method': 'normal', 'iparm': {1: 13}}
FIRST_YEAR = ('RAKTIEDRIFT* RVIRKDISK* FVIRKDISK* RFINACCELPREM* DFINFRIKTION* SFVIRKDISK* '
              'SDOPSLAGOMK2DNLLAG*')


class Recorder:
    """A factorizer for ModelFlow that writes J and every right hand side to a folder."""

    def __init__(self, folder, spec):
        import modelfactorize
        self.folder = Path(folder)
        self.inner = modelfactorize.from_spec(spec)
        self.spec = spec
        self.nfactor = 0
        self.nsolve = 0
        self.log = []
        self.__name__ = f'Recorder({modelfactorize.describe(spec)})'
        self.takes_verbose = True

    def __call__(self, A, verbose=False):
        self.nfactor += 1
        if self.nfactor > 1:
            raise RuntimeError('a second factorization: the export expects one Jacobian per case')
        A = sps.csr_matrix(A)
        sps.save_npz(self.folder / 'A.npz', A, compressed=True)
        t0 = time.perf_counter()
        inner_solve = self.inner(A, verbose)
        self.factor_seconds = time.perf_counter() - t0
        self.factor_info = getattr(inner_solve.__self__, 'info', '')

        def solve(b):
            b = np.asarray(b, dtype=np.float64)
            t0 = time.perf_counter()
            x = inner_solve(b)
            seconds = time.perf_counter() - t0
            k = self.nsolve
            np.save(self.folder / f'b_{k:02d}.npy', b)
            np.save(self.folder / f'x_{k:02d}.npy', x)
            relres, refinements = getattr(inner_solve.__self__, 'last', (np.nan, -1))
            self.log.append({'rhs': k, 'seconds': round(seconds, 3), 'relres': float(relres),
                             'refinements': int(refinements), 'norm_b': float(np.abs(b).max())})
            self.nsolve += 1
            return x
        solve.__self__ = inner_solve.__self__ if hasattr(inner_solve, '__self__') else None
        return solve


def matrix_stats(A):
    A = A.tocsr()
    d = A.diagonal()
    return {'n': A.shape[0], 'nnz': int(A.nnz), 'zero_diagonals': int((d == 0).sum()),
            'nnz_per_row_max': int(np.diff(A.indptr).max())}


def export(mmakro, baseline, case):
    first, last = CASES[case]
    folder = OUT / case
    folder.mkdir(parents=True, exist_ok=True)
    for f in folder.glob('*.npy'):
        f.unlink()
    rec = Recorder(folder, REFERENCE)
    opts = dict(solver='newtonstack_ng', forcenum=False, factorizer=rec, ljit=True, silent=1,
                reset_options=True)

    t0 = time.perf_counter()
    base = mmakro(baseline, first, last, **opts)
    alt = base.upd(f'<{first} {last}> QR__OFF * 1.01')
    alt = mmakro.fix(alt, FIRST_YEAR, first, first, silent=1)
    res = mmakro(alt, first, last, **opts)
    seconds = time.perf_counter() - t0

    diff = mmakro.ng_newton_diff_stack
    periods = list(range(first, last + 1))
    A = sps.load_npz(folder / 'A.npz')
    meta = {
        'case': case,
        'model': 'MAKRO 2026-September (DREAM, github DREAM-DK/MAKRO) translated to ModelFlow',
        'problem': 'stacked Newton Jacobian J of the simultaneous core, all periods at once; J x = b per Newton step',
        'experiment': f'baseline solve, then public purchases QR__OFF +1 % {first}-{last}',
        'periods': [first, last],
        'n_per_period': len(diff.endovar),
        'layout': 'rows and columns are period-major: index = period_number * n_per_period + position in the lists',
        **matrix_stats(A),
        'rhs': rec.log,
        'reference_method': rec.__name__,
        'reference_factor_seconds': round(rec.factor_seconds, 2),
        'reference_factor_info': rec.factor_info,
        'reference_machine': machine(),
        'total_seconds': round(seconds, 1),
        'max_change_QR__OFF': float((res.loc[first:last, 'QR__OFF'] / base.loc[first:last, 'QR__OFF'] - 1).max()),
    }
    (folder / 'meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    (folder / 'names.json').write_text(json.dumps({'equations': list(diff.endovar),
                                                    'unknowns': list(diff.declared_endo_list),
                                                    'periods': periods}), encoding='utf-8')
    print(f'{case}: n={meta["n"]:,} nnz={meta["nnz"]:,} rhs={len(rec.log)} '
          f'factor {rec.factor_seconds:.1f} s, all {seconds:.1f} s', flush=True)
    for r in rec.log:
        print('   ', r, flush=True)


def machine():
    import platform
    out = {'platform': platform.platform(), 'processor': platform.processor(), 'python': platform.python_version()}
    try:
        import psutil
        out['memory_gb'] = round(psutil.virtual_memory().total / 2**30)
        out['cores_physical'] = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    return out


def main(args):
    cases = list(CASES) if not args or args == ['all'] else args
    unknown = set(cases) - set(CASES)
    if unknown:
        sys.exit(f'unknown case(s) {sorted(unknown)}, use {", ".join(CASES)} or all')
    os.chdir(MAKRO)                          # the jit code of the model lives in makro26\modelsource
    sys.path.insert(0, str(MAKRO))
    from modelclass import model
    mmakro, baseline = model.modelload(str(MAKRO / 'data' / 'makro26_baseline.pcimz'))
    for case in cases:
        export(mmakro, baseline, case)


if __name__ == '__main__':
    main(sys.argv[1:])

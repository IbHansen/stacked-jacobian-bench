# -*- coding: utf-8 -*-
"""MUMPS through python-mumps, in this process (the ModelFlow helper-process variant is not used here)."""
import mumps  # noqa: F401  (fail at import when python-mumps is missing)

from ..modelfactorize import MumpsSolver


def mumps(A, verbose=False, refine_tol=None, krylov=False):
    """MUMPS LU with METIS ordering, refined against J"""
    s = MumpsSolver.__new__(MumpsSolver)
    s.krylov = bool(krylov)
    if refine_tol is not None:
        s.refine_tol = float(refine_tol)
    s.__init__(A, verbose)
    return s.solve

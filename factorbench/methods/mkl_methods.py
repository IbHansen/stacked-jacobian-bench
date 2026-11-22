# -*- coding: utf-8 -*-
"""MKL PARDISO through pypardiso: exactly the code ModelFlow runs (factorbench/modelfactorize.py).

Options: iparm={1: 13} (0-based MKL C numbering; 13 = parallel nested dissection + oneMKL 2026's faster
factorization), krylov=True (GMRES instead of repeated correction), refine_tol=1e-6 (what MAKRO uses).
"""
from .. import modelfactorize as mf

mf._pypardiso()            # fail at import when pypardiso is missing, so the registry can say which extra to install


def pardiso(A, verbose=False, iparm=None, krylov=False, refine_tol=None):
    """MKL PARDISO LU of J (static pivoting), refined against J"""
    return mf.pardiso(iparm, krylov, refine_tol)(A, verbose)


def normal(A, verbose=False, iparm=None, krylov=False, refine_tol=None):
    """normal equations J'J with a PARDISO Cholesky, refined against J"""
    return mf.normal(iparm, krylov, refine_tol)(A, verbose)

# -*- coding: utf-8 -*-
"""Methods that need scipy only."""
import scipy.sparse.linalg as spla

from ..modelfactorize import RefinedSolver


class SuperLUSolver(RefinedSolver):
    """scipy's SuperLU, refined against J. permc_spec: COLAMD (default), NATURAL, MMD_ATA, MMD_AT_PLUS_A."""
    name = 'superlu'

    def __init__(self, A, verbose=False, permc_spec='COLAMD', refine_tol=None, krylov=False):
        self.permc_spec = permc_spec
        self.krylov = bool(krylov)
        if refine_tol is not None:
            self.refine_tol = float(refine_tol)
        super().__init__(A, verbose)

    def _factorize(self):
        self.lu = spla.splu(self.A.tocsc(), permc_spec=self.permc_spec)
        self.info = f', permc_spec {self.permc_spec}, fill L+U {self.lu.L.nnz + self.lu.U.nnz:,}'

    def _apply(self, r):
        return self.lu.solve(r)

    def _release(self):
        self.__dict__.pop('lu', None)


def superlu(A, verbose=False, **options):
    """scipy SuperLU (LU with partial pivoting), refined against J"""
    return SuperLUSolver(A, verbose, **options).solve

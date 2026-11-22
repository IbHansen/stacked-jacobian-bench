# -*- coding: utf-8 -*-
"""
template.py - how to add a factorization method. Copy this file to factorbench/methods/<yours>.py, rename, and
it is found by name:  python -m factorbench run --case makro100 --method mine

A method is a function  factorize(A, verbose=False, **options) -> solve,  solve(b) -> x  (ModelFlow's
``factorizer=`` contract, so the same function can be given to ModelFlow as factorizer=...).

The simplest way is a subclass of RefinedSolver: implement _factorize (factorize self.A, a scipy CSR matrix) and
_apply (an approximate solution of J x = r). The base class refines against J until the relative residual
(max norm) is below refine_tol (1e-11 by default, MAKRO uses 1e-6), or uses GMRES with the factorization as
preconditioner (krylov=True), and fails clearly when it can not get below 1e-2.

This example is plain scipy SuperLU with the NATURAL ordering. It only shows the mechanics: on MAKRO it is far
too slow (a 10-year version of the problem ran more than 13 minutes and 8.5 GB without finishing; SuperLU with COLAMD takes 230 s, the
'normal' method 5 s). The template registers nothing; your copy does through METHODS at the bottom.
"""
import scipy.sparse.linalg as spla

from ..modelfactorize import RefinedSolver


class MySolver(RefinedSolver):
    name = 'mine'

    def __init__(self, A, verbose=False, refine_tol=None, krylov=False):
        self.krylov = bool(krylov)
        if refine_tol is not None:
            self.refine_tol = float(refine_tol)
        super().__init__(A, verbose)          # calls _factorize

    def _factorize(self):
        self.lu = spla.splu(self.A.tocsc(), permc_spec='NATURAL')
        self.info = f', fill {self.lu.L.nnz + self.lu.U.nnz:,}'    # shown in the benchmark output

    def _apply(self, r):
        return self.lu.solve(r)

    def _release(self):                       # free memory the garbage collector does not see (GPU, MKL)
        self.__dict__.pop('lu', None)


def mine(A, verbose=False, **options):
    """example: SuperLU with NATURAL ordering (template.py)"""
    return MySolver(A, verbose, **options).solve


# METHODS = {'mine': mine}            # uncomment in your copy: the name -> function the registry finds

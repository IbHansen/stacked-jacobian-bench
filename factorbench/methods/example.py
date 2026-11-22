# -*- coding: utf-8 -*-
"""example.py - a complete method: the normal equations J'J x = J'b with a PARDISO Cholesky, refined against J.

Run it:   python -m factorbench run --case makro100 --method example
Your own: copy this file to factorbench/methods/mine.py, change the last line to METHODS = {'mine': example}
          and replace the lines marked '<- your' with your library. See README.md, Create a solver.
"""
import pypardiso
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from factorbench.modelfactorize import RefinedSolver


class ExampleSolver(RefinedSolver):
    name = 'example'

    def _factorize(self):                     # called once; self.A is the Jacobian J (scipy CSR), unscaled
        # 1. Scaling: Js = diag(dr) A diag(dc), rows to largest element 1, then columns to length 1.
        #    Without it the refinement stalls at about 1e-7 on MAKRO.
        self.dr = 1 / abs(self.A).max(axis=1).toarray().ravel()
        Jr = sps.diags(self.dr) @ self.A                            # rows scaled
        self.dc = 1 / spla.norm(Jr, axis=0)
        Js = (Jr @ sps.diags(self.dc)).tocsr()                      # rows and columns scaled

        # 2. The matrix to factorize: the normal equations Js'Js (PARDISO wants only the upper triangle)
        self.JsT = Js.T.tocsr()
        self.JsTJs = sps.triu(self.JsT @ Js, format='csr')

        # 3. PARDISO: factorize Js'Js once
        self.pardiso = pypardiso.PyPardisoSolver(mtype=2)           # <- your library: 2 = symmetric pos. definite
        self.pardiso.factorize(self.JsTJs)                          # <- your factorization

    def _apply(self, r):                      # an approximate x for J x = r, using the factors
        # Js'Js y = Js' (dr r) with PARDISO, then undo the column scaling: x = dc y
        y = self.pardiso.solve(self.JsTJs, self.JsT @ (self.dr * r))   # <- your solve
        return self.dc * y

    def _release(self):                       # free memory Python does not see (here MKL's)
        self.pardiso.free_memory(everything=True)


def example(A, verbose=False):
    """the example of README.md: normal equations J'J, PARDISO Cholesky, refined against J"""
    return ExampleSolver(A, verbose).solve


METHODS = {'example': example}

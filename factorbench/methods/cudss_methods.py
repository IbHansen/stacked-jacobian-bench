# -*- coding: utf-8 -*-
"""NVIDIA cuDSS (sparse direct solver on the GPU) through nvmath-python  (pip install -e .[gpu]).

Written against nvmath-python 1.0.1 WITHOUT access to an NVIDIA GPU: the API calls follow the nvmath source, but
the code has not run. NVIDIA: please correct what needs correcting.

Two methods, both refined against J on the CPU by factorbench's RefinedSolver (so the accuracy is measured the
same way for every method):

  cudss          LU of J (matrix type GENERAL).
  cudss_normal   Cholesky of the normal equations J'J (matrix type SPD, upper triangle), J scaled first - the same
                 formulation as the 'normal' method (PARDISO) that MAKRO uses now.

Options (``--opt name=value`` on the command line):
  hybrid=True          keep the factors in host memory (cuDSS hybrid memory mode) when they do not fit on the GPU
  device_limit='40GiB' device memory limit in hybrid mode
  reordering='NESTED_DISSECTION' | 'AMD' | 'COLAMD' | 'BTF_COLAMD' | 'DEFAULT'
  matching='MAX_DIAG_PRODUCT' | ...   (cudss only) a matching (MC64-like) before the ordering - J has many zero
                       diagonals (MAKRO: 26,300 of 1,547,200 rows, about 1/60), which static pivoting alone handles badly
  pivot_eps=1e-13, pivot_type='PIVOT_GLOBAL_ROW' | ...
  ir_steps=0           cuDSS's own iterative refinement steps (factorbench refines anyway)
  refine_tol, krylov   as for the other methods

Note on the data path: J and every b are numpy/scipy (host) arrays, so each solve copies b to the GPU and x back.
For the Newton use that is the realistic case (ModelFlow evaluates the model on the CPU).
"""
import numpy as np
import scipy.sparse as sps

import nvmath  # noqa: F401  (fail at import when nvmath-python is missing)
from nvmath.bindings import cudss
from nvmath.sparse import advanced as nva

from ..modelfactorize import RefinedSolver


def _scaled(A):
    """J scaled as in the 'normal' method: rows to largest element 1, then columns to norm 1.
    Returns (dr, dc, AsT) with As = diag(dr) J diag(dc)."""
    rowmax = np.asarray(abs(A).max(axis=1).todense()).ravel()
    dr = 1 / rowmax
    As = sps.diags(dr) @ A
    colnorm = np.sqrt(np.asarray(As.multiply(As).sum(axis=0)).ravel())
    dc = 1 / colnorm
    AsT = (As @ sps.diags(dc)).T.tocsr()
    return dr, dc, AsT


class CudssSolver(RefinedSolver):
    name = 'cudss'
    normal = False

    def __init__(self, A, verbose=False, hybrid=False, device_limit=None, reordering=None, matching=None,
                 pivot_eps=None, pivot_type=None, ir_steps=None, refine_tol=None, krylov=False):
        self.opts = dict(hybrid=hybrid, device_limit=device_limit, reordering=reordering, matching=matching,
                         pivot_eps=pivot_eps, pivot_type=pivot_type, ir_steps=ir_steps)
        self.krylov = bool(krylov)
        if refine_tol is not None:
            self.refine_tol = float(refine_tol)
        super().__init__(A, verbose)

    def _factorize(self):
        o = self.opts
        if self.normal:
            self.dr, self.dc, self.AsT = _scaled(self.A)
            M = sps.triu(self.AsT @ self.AsT.T, format='csr')
            options = nva.DirectSolverOptions(sparse_system_type=nva.DirectSolverMatrixType.SPD,
                                              sparse_system_view=nva.DirectSolverMatrixViewType.UPPER)
        else:
            M = self.A
            options = nva.DirectSolverOptions(sparse_system_type=nva.DirectSolverMatrixType.GENERAL)
        M = sps.csr_matrix(M)
        M.sort_indices()
        M.indptr = M.indptr.astype(np.int32)
        M.indices = M.indices.astype(np.int32)
        self.M = M
        execution = nva.ExecutionCUDA(hybrid_memory_mode_options=nva.HybridMemoryModeOptions(
            hybrid_memory_mode=bool(o['hybrid']), hybrid_device_memory_limit=o['device_limit']))
        self.solver = nva.DirectSolver(M, np.zeros(M.shape[0]), options=options, execution=execution)
        pc, fc, sc = self.solver.plan_config, self.solver.factorization_config, self.solver.solution_config
        if o['reordering']:
            pc.reordering_algorithm = getattr(cudss.ReorderingAlg, o['reordering'])
        if o['matching'] and not self.normal:
            pc.matching_algorithm = getattr(cudss.MatchingAlg, o['matching'])
        if o['pivot_type']:
            pc.pivot_type = getattr(cudss.PivotType, o['pivot_type'])
        if o['pivot_eps'] is not None:
            fc.pivot_eps = float(o['pivot_eps'])
        if o['ir_steps'] is not None:
            sc.ir_num_steps = int(o['ir_steps'])
        import time
        t0 = time.perf_counter()
        plan_info = self.solver.plan()
        t1 = time.perf_counter()
        fac_info = self.solver.factorize()
        t2 = time.perf_counter()
        details = [f'plan {t1 - t0:.1f} s, factorize {t2 - t1:.1f} s']
        for label, get in (('LU non-zeros', lambda: fac_info.lu_nnz), ('pivots perturbed', lambda: fac_info.npivots),
                           ('memory estimates', lambda: plan_info.memory_estimates)):
            try:
                details.append(f'{label} {get()}')
            except Exception:
                pass
        if self.normal:
            details.insert(0, f"J'J {2 * M.nnz - M.shape[0]:,} non-zeros")
        self.info = ', ' + ', '.join(details) + ''.join(f', {k}={v}' for k, v in o.items() if v)

    def _gpu_solve(self, r):
        self.solver.reset_operands(b=np.ascontiguousarray(r, dtype=np.float64))
        return np.asarray(self.solver.solve())

    def _apply(self, r):
        if self.normal:
            return self.dc * self._gpu_solve(self.AsT @ (self.dr * r))
        return self._gpu_solve(r)

    def _release(self):
        s = self.__dict__.pop('solver', None)
        if s is not None:
            s.free()
        self.__dict__.pop('M', None)


class CudssNormalSolver(CudssSolver):
    name = 'cudss_normal'
    normal = True


def cudss_lu(A, verbose=False, **options):
    """NVIDIA cuDSS LU of J on the GPU, refined against J"""
    return CudssSolver(A, verbose, **options).solve


def cudss_normal(A, verbose=False, **options):
    """NVIDIA cuDSS Cholesky of J'J on the GPU, refined against J"""
    return CudssNormalSolver(A, verbose, **options).solve

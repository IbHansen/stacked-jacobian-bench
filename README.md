# factorbench: factorizing a 1.5-million-unknown stacked Newton Jacobian

A standalone test of the linear solve inside the Newton solver of a large forward-looking economic model, so other
factorization routines, and GPU solvers in particular, can be tried on the real matrix without the model itself.

The matrix comes from **MAKRO**, the Danish macroeconomic model of DREAM (2026-September release,
[github.com/DREAM-DK/MAKRO](https://github.com/DREAM-DK/MAKRO), MIT licence), translated to
[ModelFlow](https://github.com/IbHansen/ModelFlow2). A forward-looking model has to be solved for all years at
once: the 100 years 2030-2129 are stacked into one nonlinear system, and each Newton step solves

    J x = b,    J = the Jacobian of all equations of all years with respect to all unknowns of all years

## The problem

| | |
|---|---|
| unknowns n | 1,547,200 (15,472 per year x 100 years) |
| non-zeros of J | 7,653,248 (about 5 per row) |
| structure | period-major block band: lags 1-4 years and a lead of 1 year (88 % of the entries within a year), plus about 200 entries that tie every year to the terminal year |
| zero diagonals | 26,300 |
| right-hand sides | 8: the residuals of the Newton iterations of one experiment (public purchases +1 %), in order |

Why it is hard:
* LU with **static pivoting** (MKL PARDISO on J) factorizes in 15 s and 6 GB, but the solution is wrong (residual 1e2-1e4 after refinement): the zero diagonals and the saddle-path structure defeat it.
* **SuperLU** (partial pivoting) works up to about 20 years; beyond that the fill-in makes it fail.
* GMRES with block Gauss-Seidel preconditioning does not converge (saddle path).
* What works now: the **normal equations** J'J x = J'b (J scaled first) with a PARDISO Cholesky factorization,
  refined against J itself. MUMPS (LU with real pivoting, METIS) also works but is about 2.5 times slower.
* J is badly conditioned. A relative residual of 1e-12 still leaves up to 1e-6 relative difference from the reference
  solution.

The pattern that matters is **factorize once, solve many times**: one factorization per experiment, then one solve
(with refinements) per Newton iteration. On this laptop (AMD Ryzen AI 9 HX 370, 12 cores, 64 GB) with the method
MAKRO uses now (`normal`, `iparm={1: 13}`, `refine_tol=1e-6`), one experiment takes:

| | seconds |
|---|---|
| factorization of J'J (85.7 million non-zeros) | 34 |
| 8 solves incl. refinements (each application of the factors is about 1.3 s) | 19 |
| total, peak memory 17 GB | 54 |

The solves are limited by memory bandwidth (16 GB of factors read per application). That is where faster memory,
a GPU for example, could pay off.

## Install (pip only)

Python 3.10 or newer. This installs what is needed to run the reference method (`normal`: numpy, scipy, psutil and
MKL PARDISO) and downloads the matrix (150 MB zip, 225 MB unpacked, checksum-verified).

Windows:

```
git clone https://github.com/IbHansen/stacked-jacobian-bench
cd stacked-jacobian-bench
python -m venv .venv
.venv\Scripts\activate
pip install -e .[pardiso]
python get_data.py
```

Linux / macOS:

```
git clone https://github.com/IbHansen/stacked-jacobian-bench
cd stacked-jacobian-bench
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[pardiso]
python get_data.py
```

Then add what your own method needs. For the NVIDIA cuDSS methods (CUDA 12) that is `pip install -e .[gpu]`,
or `.[gpu13]` for CUDA 13. `setup_env.bat` and `setup_env.sh` do the same steps, after the clone (`setup_env.sh gpu`
adds the GPU packages).

## Create a solver

The repository has a complete working method in one file, `factorbench/methods/example.py`. Run it:

```
python -m factorbench run --case makro100 --method example
```

It solves the full problem: the normal equations J'J with a PARDISO Cholesky, refined against J (the same as the
`normal` method). Here is the whole file:

```python
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
```

To try your own solver:

1. Copy the file to `factorbench/methods/mine.py`.
2. In its last line, change the name the benchmark knows it by: `METHODS = {'mine': example}`.
3. Replace the three lines marked `# <- your ...` with your library: create it, factorize, solve. Step 1 (the
   scaling) and step 2 (the normal equations) belong to this method: keep them, or drop them if yours factorizes J
   itself.
4. Run it and compare with the reference: see Run below.

What the parts do:

* `_factorize` is called once, with J in `self.A` as a scipy CSR matrix.
* `_apply(r)` is one solve with the factors.
* `_release` frees memory Python cannot see (MKL's here, GPU memory for a GPU library).
* `RefinedSolver` provides the rest. Its `solve(b)` calls your `_apply`, computes `b - J x` with the original J, and
  repeats with the residual until it is below 1e-11 (`--opt refine_tol=1e-6` loosens it). That is why an
  approximate factorization such as J'J still gives an accurate x.

The function `example(A, verbose=False)` returning a `solve` is ModelFlow's `factorizer=` contract, so the same
function works there: `model(..., factorizer=example)`. `cudss_methods.py` is a larger example with options.

## Run

Run the reference method, as MAKRO uses it, then your own, then compare:

```
python -m factorbench run --case makro100 --method normal --opt "iparm={1: 13}" --opt refine_tol=1e-6
python -m factorbench run --case makro100 --method mine
python -m factorbench compare
```

`python -m factorbench list` shows all methods and the downloaded cases.

Each run factorizes J once and solves the 8 right-hand sides in order. It reports for each:

* `factor_s`: time of the factorization (ordering, symbolic and numeric, including any copy to a device)
* `solve_s`: time of each solve, including refinement and copies
* `relres = max|b - J x| / max|b|`: computed by the benchmark, independent of the method. Newton needs about 1e-6
* `err_ref`: the distance from the reference solution (see the conditioning remark above)
* the peak memory of the process, and the machine and package versions

The result goes to `results/<case>__<method>__<host>__<time>.json`. To share a result, send the file or open a
pull request that adds it to `results/published/`.

## The methods

| method | what it does | library | read more |
|---|---|---|---|
| `superlu` | LU of J with partial pivoting | SuperLU through scipy | [scipy splu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html), [SuperLU](https://portal.nersc.gov/project/sparse/superlu/) |
| `pardiso` | LU of J with static pivoting (small pivots are perturbed, not swapped) | Intel oneMKL PARDISO through pypardiso | [oneMKL PARDISO](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/onemkl-pardiso-parallel-direct-sparse-solver-iface.html), [pypardiso](https://github.com/haasad/PyPardiso) |
| `normal` | Cholesky of the normal equations J'J (J scaled first), then refinement against J. What MAKRO uses now | oneMKL PARDISO through pypardiso | [normal equations](https://en.wikipedia.org/wiki/Linear_least_squares), [Cholesky](https://en.wikipedia.org/wiki/Cholesky_decomposition) |
| `mumps` | LU of J with threshold pivoting and METIS ordering | MUMPS through python-mumps | [MUMPS](https://mumps-solver.org/), [python-mumps](https://gitlab.kwant-project.org/kwant/python-mumps) |
| `cudss` | LU of J on an NVIDIA GPU | NVIDIA cuDSS through nvmath-python | [cuDSS](https://docs.nvidia.com/cuda/cudss/), [nvmath-python](https://docs.nvidia.com/cuda/nvmath-python/latest/) |
| `cudss_normal` | Cholesky of J'J on an NVIDIA GPU, refinement against J | NVIDIA cuDSS through nvmath-python | as above |

All of them solve each right-hand side with [iterative refinement](https://en.wikipedia.org/wiki/Iterative_refinement)
against J. With `--opt krylov=True` they use [GMRES](https://en.wikipedia.org/wiki/Generalized_minimal_residual_method)
instead, with the factorization as preconditioner. See [LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition)
for the background.

## Files

| | |
|---|---|
| `factorbench/modelfactorize.py` | verbatim copy of ModelFlow's `modelfactorize.py`: the reference methods (normal, pardiso, mumps) and `RefinedSolver` |
| `factorbench/methods/` | the methods: `scipy_methods`, `mkl_methods`, `mumps_methods`, `cudss_methods`, `example` (the example above) |
| `factorbench/bench.py` | the benchmark and the command line |
| `get_data.py`, `data_manifest.json` | download of the matrix from the GitHub release |
| `tools/to_mtx.py` | the case in Matrix Market format, for solvers outside Python |
| `tools/make_release.py` | packs data/ for a release |
| `export/export_makro.py` | how the matrix was made (needs ModelFlow and the MAKRO translation; not needed to run anything) |
| `NVIDIA.md` | the questions for a GPU test |

## The data

`data/makro100/`: `A.npz` (J, scipy CSR), `b_00.npy` ... `b_07.npy`, the reference solutions `x_00.npy` ...
(normal equations + PARDISO, refined to about 1e-12), `meta.json` (sizes, reference timings), and `names.json`
(the equation and unknown names of one year; index = year number x 15,472 + position).

The files are NumPy's binary format ([`.npy`](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html),
one array per file) and scipy's sparse format (`A.npz`: a zip of the CSR arrays `data`, `indices`, `indptr`, read
with `scipy.sparse.load_npz`). For solvers outside Python, `python tools/to_mtx.py makro100` writes the same case in
Matrix Market format (`.mtx`) to `data/makro100/mtx/`: 0.28 GB for A, 0.83 GB with all vectors.

The matrix is derived from MAKRO (Copyright (c) 2026 DREAM, MIT licence). Code: MIT licence.

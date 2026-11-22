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

Python 3.10 or newer.

```
git clone https://github.com/IbHansen/stacked-jacobian-bench
cd stacked-jacobian-bench
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -e .                  # numpy, scipy, psutil: the method 'superlu'
pip install -e .[pardiso]         # + MKL PARDISO: the methods 'normal' and 'pardiso' (the reference)
pip install -e .[gpu]             # + NVIDIA cuDSS through nvmath-python: 'cudss' and 'cudss_normal' (CUDA 12)
pip install -e .[mumps]           # + MUMPS (pip wheels for Linux/macOS)
python get_data.py                # downloads the matrix: 150 MB zip, 225 MB unpacked, checksum-verified
```

`setup_env.bat` / `setup_env.sh` run these steps (`setup_env.sh gpu` adds the GPU packages).

## Run

```
python -m factorbench list                                  # methods and downloaded cases
python -m factorbench run --case makro100 --method normal --opt "iparm={1: 13}" --opt refine_tol=1e-6
python -m factorbench run --case makro100 --method cudss_normal --opt hybrid=True
python -m factorbench run --case makro100 --method cudss --opt matching=MAX_DIAG_PRODUCT
python -m factorbench compare                               # table of all results/*.json
```

Each run factorizes J once and solves the 8 right-hand sides in order. It reports for each:

* `factor_s`: time of the factorization (ordering, symbolic and numeric, including any copy to a device)
* `solve_s`: time of each solve, including refinement and copies
* `relres = max|b - J x| / max|b|`: computed by the benchmark, independent of the method. Newton needs about 1e-6
* `err_ref`: the distance from the reference solution (see the conditioning remark above)
* the peak memory of the process, and the machine and package versions

The result goes to `results/<case>__<method>__<host>__<time>.json`. To share a result, send the file or open a
pull request that adds it to `results/published/`.

## Adding a method

A method is a function with ModelFlow's `factorizer=` contract, so anything that works here can be passed to
ModelFlow unchanged:

```python
def factorize(A, verbose=False, **options):      # A: scipy CSR
    ...                                          # factorize once
    return solve                                 # solve(b) -> x, called once per Newton iteration
```

The easy way is a subclass of `RefinedSolver` (`factorbench/modelfactorize.py`). You implement `_factorize()` and
`_apply(r)`, and you get iterative refinement against J, GMRES as an alternative (`krylov=True`), stall detection
and the accuracy checks for free:

```python
from factorbench.modelfactorize import RefinedSolver

class MySolver(RefinedSolver):
    name = 'mine'
    def _factorize(self):              # self.A is J (CSR)
        self.f = my_library.factorize(self.A)
    def _apply(self, r):               # an approximate solution of J x = r
        return self.f.solve(r)
    def _release(self):                # free GPU / library memory
        self.f.free()

def mine(A, verbose=False, **options):
    return MySolver(A, verbose, **options).solve

METHODS = {'mine': mine}
```

To add it, save it as `factorbench/methods/mine.py` and run `--method mine`. Without copying it into the package,
use `--method mymodule:mine`. See `factorbench/methods/template.py` and `cudss_methods.py`.

## Files

| | |
|---|---|
| `factorbench/modelfactorize.py` | verbatim copy of ModelFlow's `modelfactorize.py`: the reference methods (normal, pardiso, mumps) and `RefinedSolver` |
| `factorbench/methods/` | the methods: `scipy_methods`, `mkl_methods`, `mumps_methods`, `cudss_methods`, `template` |
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

The matrix is derived from MAKRO (Copyright (c) 2026 DREAM, MIT licence). Code: MIT licence.

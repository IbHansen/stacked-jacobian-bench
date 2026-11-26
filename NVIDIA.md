# Can an NVIDIA GPU speed up the Newton step of a large forward-looking model?

## The problem in one paragraph

MAKRO, the Danish macroeconomic model (DREAM), is solved for 100 years at once by a stacked Newton method. Each
Newton step solves `J x = b` with a sparse, unsymmetric, badly conditioned J. J has n = 1,547,200 unknowns, 7.65
million non-zeros, 26,300 zero diagonals, and a block-band structure over the years (lags 1-4, lead 1) plus a few
entries to the terminal year. One experiment is one factorization followed by 6-8 solves (one per Newton
iteration). A policy analysis runs tens of experiments. The repository has J, the 8 right-hand sides of a real
experiment in order, and reference solutions.

## What works today (CPU)

AMD Ryzen AI 9 HX 370 (12 cores), 64 GB, oneMKL 2026.2 through pypardiso:

| method | factorization | 8 solves | total | peak memory |
|---|---|---|---|---|
| normal equations J'J (scaled), PARDISO Cholesky, `iparm[1]=13`, refined against J to 1e-6 | 34 s | 19 s | 54 s | 17 GB |
| the same, refined to 1e-11 (ModelFlow's default) | 50 s | 55 s | 105 s | 17.5 GB |
| PARDISO LU of J (static pivoting), `iparm[1]=13` | 15 s | fails: residual 1e2-1e4 after refinement | | 6 GB |
| SuperLU | fails beyond about 20 years (fill) | | | |
| MUMPS LU (METIS, real pivoting) | works, about 2.5 times slower than the normal equations | | | |

J'J has 85.7 million non-zeros, and its Cholesky factor takes about 16 GB. **The solves dominate once the
factorization is done.** Each application of the factors takes about 1.3 s and is memory-bandwidth bound. The
refinement needs 0-3 applications per Newton step at 1e-6 and 3-8 at 1e-11.

## Hypothesis

GPU memory bandwidth (H100/H200 about 3-5 TB/s against about 0.1 TB/s here) should speed up the triangular
solves the most, and the factorization too. Two ways to get there:

1. **`cudss_normal`**: the same formulation as today (Cholesky of the scaled J'J, SPD) on the GPU. The 16 GB factor
   fits on an 80 GB card. Otherwise use hybrid memory mode.
2. **`cudss`**: LU of J itself, avoiding the squared condition number. On the CPU the LU is much cheaper than
   J'J (PARDISO: 15 s and 6 GB against 34 s and 17 GB), but static pivoting makes it wrong. Can cuDSS's matching
   (`MatchingAlg`) and pivoting options make it work? If so, this is the most promising route.

## What we would like to know

On an H100, H200 or GH200, if possible:

* `factor_s`, the time of each solve, and the total for both methods.
* Whether `cudss` (LU) reaches the accuracy with matching and/or pivoting options, and with which settings.
* GPU memory use, and whether hybrid mode is needed.
* Whether GH200's unified memory changes the picture, since the model evaluation itself stays on the CPU and b and x
  cross the bus once per Newton iteration.
* Anything we should do differently: ordering, the formulation, cuDSS iterative refinement instead of ours, batched
  or multi-GPU.

## Install

Linux, Python 3.10 or newer. The reference method needs MKL PARDISO, and the cuDSS methods need nvmath-python:

```
git clone https://github.com/IbHansen/stacked-jacobian-bench
cd stacked-jacobian-bench
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[pardiso]
pip install -e .[gpu]
python get_data.py
```

For CUDA 13, use `pip install -e .[gpu13]` instead of `.[gpu]`. `get_data.py` downloads the matrix (150 MB) and
checks it.

## The cuDSS methods

Two methods are ready in `factorbench/methods/cudss_methods.py`:

* `cudss`: LU of J.
* `cudss_normal`: Cholesky of the scaled normal equations J'J, the same formulation as the reference.

In both, cuDSS factorizes once on the GPU, and each solve is one forward and back substitution there. The benchmark
then refines each solution against the original J on the CPU, the same way for every method, so the accuracy
figures are comparable.

**Note:** the adapter was written against nvmath-python 1.0.1 without access to an NVIDIA GPU, so it has not run.
Please correct it as needed. To try another formulation or library, README.md, "Create a solver", has a complete
example file. A method is one function, `solve = mine(A)`, then `x = solve(b)` for each right-hand side.

## Run

The CPU reference as MAKRO uses it, the two cuDSS methods, then the comparison:

```
python -m factorbench run --case makro100 --method normal --opt "iparm={1: 13}" --opt refine_tol=1e-6
python -m factorbench run --case makro100 --method cudss_normal --opt refine_tol=1e-6
python -m factorbench run --case makro100 --method cudss --opt matching=MAX_DIAG_PRODUCT --opt refine_tol=1e-6
python -m factorbench compare
```

Options are passed as `--opt name=value`. The cuDSS options are listed at the top of `cudss_methods.py`:
`hybrid`, `device_limit`, `reordering`, `matching`, `pivot_type`, `pivot_eps`, `ir_steps`, `refine_tol`,
`krylov`. The benchmark measures the accuracy itself, as `max|b - J x| / max|b|`; Newton needs about 1e-6.

Results are written to `results/*.json`. We would be glad to get those files back, with any changes to the adapter.

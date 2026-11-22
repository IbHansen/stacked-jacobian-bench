# -*- coding: utf-8 -*-
"""
registry.py - the factorization methods by name.

A method is a function with ModelFlow's ``factorizer=`` contract:

    factorize(A, verbose=False, **options) -> solve      and  solve(b) -> x

A is J (scipy CSR), b and x are numpy float64 vectors. ``solve`` is called several times with the same
factorization (one call per Newton iteration). Optional, used by the benchmark when present:

    solve.__self__.info     a text on the factorization (ordering, fill, memory, device ...)
    solve.__self__.last     (relative residual, refinements) of the latest solve
    solve.__self__.release()  free the factorization (GPU and MKL memory the garbage collector does not see)

The easy way is a subclass of ``RefinedSolver`` (factorbench/modelfactorize.py): implement ``_factorize()``
(factorize self.A) and ``_apply(r)`` (the approximate solution of J x = r). Refinement against J, GMRES
(krylov=True), stall detection and the accuracy checks come with it. See methods/example.py.

Methods are found in three places:
  * the built-in methods below (each in factorbench/methods/<module>.py, imported only when used)
  * every file factorbench/methods/*.py with a dict METHODS = {'name': function}
  * on the command line as  module:function  (a module on the python path, e.g. mysolver:factorize)
"""
import importlib
import pkgutil

# name -> (module in factorbench.methods, function, description, the pip extra it needs)
BUILTIN = {
    'superlu': ('scipy_methods', 'superlu', 'scipy SuperLU (LU with partial pivoting), refined', None),
    'pardiso': ('mkl_methods', 'pardiso', 'MKL PARDISO LU of J (static pivoting), refined', 'pardiso'),
    'normal': ('mkl_methods', 'normal', "normal equations J'J, PARDISO Cholesky, refined against J "
                                        '(what MAKRO uses in ModelFlow)', 'pardiso'),
    'mumps': ('mumps_methods', 'mumps', 'MUMPS LU with METIS ordering, refined', 'mumps'),
    'cudss': ('cudss_methods', 'cudss_lu', 'NVIDIA cuDSS LU of J on the GPU (nvmath-python), refined', 'gpu'),
    'cudss_normal': ('cudss_methods', 'cudss_normal', "NVIDIA cuDSS Cholesky of J'J on the GPU, refined "
                                                       'against J', 'gpu'),
}


def _discovered():
    """METHODS dicts of the modules in factorbench/methods (other than the built-in ones)."""
    from . import methods
    builtin_modules = {m for m, *_ in BUILTIN.values()}
    found = {}
    for info in pkgutil.iter_modules(methods.__path__):
        if info.name in builtin_modules or info.name.startswith('_'):
            continue
        try:
            mod = importlib.import_module(f'{methods.__name__}.{info.name}')
        except ImportError:
            continue
        for name, fn in getattr(mod, 'METHODS', {}).items():
            found[name] = (mod.__name__, fn, (fn.__doc__ or '').strip().split('\n')[0])
    return found


def names():
    """All method names with a one-line description."""
    out = {k: v[2] for k, v in BUILTIN.items()}
    out.update({k: v[2] for k, v in _discovered().items()})
    return out


def get(name):
    """The factorize function of a method name, or of 'module:function'."""
    if ':' in name:
        module, fn = name.split(':', 1)
        return getattr(importlib.import_module(module), fn)
    if name in BUILTIN:
        module, fn, _, extra = BUILTIN[name]
        try:
            mod = importlib.import_module(f'factorbench.methods.{module}')
            return getattr(mod, fn)
        except ImportError as e:
            hint = f'  pip install -e .[{extra}]' if extra else ''
            raise ImportError(f'method {name!r} needs a package that is not installed: {e}.{hint}') from e
    found = _discovered()
    if name in found:
        return found[name][1]
    raise KeyError(f'unknown method {name!r}; known: {", ".join(names())} or module:function')

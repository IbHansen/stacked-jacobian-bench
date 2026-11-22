"""factorbench - sparse factorizations for the stacked Newton Jacobian of a large forward-looking model (MAKRO).

    from factorbench import cases, bench
    case = cases.load('makro100')
    res = bench.run(case, 'normal', {'iparm': {1: 13}})
    print(bench.report(res))
"""
from . import cases, registry  # noqa: F401

__version__ = '0.1.0'

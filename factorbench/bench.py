# -*- coding: utf-8 -*-
"""
bench.py - run a factorization method on a test case and measure it.

    python -m factorbench list                                   the methods and the downloaded cases
    python -m factorbench run --case makro100 --method normal
    python -m factorbench run --case makro100 --method normal --opt "iparm={1: 13}" --opt refine_tol=1e-6
    python -m factorbench run --case makro100 --method mysolver:factorize        (a function on the python path)
    python -m factorbench compare results                         a table of all result files

What is measured, as in a Newton solve: ONE factorization of J, then a solve for every right hand side b in turn
(the b's are the residuals of the Newton iterations of the MAKRO experiment, in order).

  factor_s      time of factorize(A)   (ordering + symbolic + numeric factorization, incl. moving J to a device)
  solve_s       time of each solve(b)  (incl. refinement and any copying)
  total_s       factor_s + all solve_s - the number to compare with the ModelFlow run
  relres        max|b - J x| / max|b|, computed here, independent of the method
  err_ref       max|x - x_ref| / max|x_ref| against the reference solution
  refinements   from the method, when it reports them
  peak_rss_gb   the peak resident memory of the process (host)

Each run writes results/<case>__<method>__<host>__<time>.json.
"""
import argparse
import ast
import datetime as dt
import gc
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import cases, registry

RESULTS = Path(__file__).resolve().parent.parent / 'results'
ACCEPT = 1e-6         # the relative residual a solve must reach to count as solved (what MAKRO's Newton uses)


def peak_rss_gb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        peak = getattr(mi, 'peak_wset', None)          # Windows
        if peak is None:
            import resource                             # Linux/macOS: ru_maxrss in kB (bytes on macOS)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak *= 1 if sys.platform == 'darwin' else 1024
        return round(peak / 2**30, 2)
    except Exception:
        return None


def _version(module):
    try:
        from importlib.metadata import version
        return version(module)
    except Exception:
        return None


def cpu_name():
    try:
        if sys.platform == 'win32':
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        if sys.platform.startswith('linux'):
            for line in Path('/proc/cpuinfo').read_text().splitlines():
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
        if sys.platform == 'darwin':
            return subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True,
                                  text=True).stdout.strip()
    except Exception:
        pass
    return platform.processor()


def machine():
    out = {'host': socket.gethostname(), 'platform': platform.platform(), 'processor': cpu_name(),
           'python': platform.python_version(),
           'packages': {p: _version(p) for p in ('numpy', 'scipy', 'pypardiso', 'mkl', 'python-mumps',
                                                 'nvmath-python', 'cupy-cuda12x', 'cupy-cuda13x')
                        if _version(p)}}
    try:
        import psutil
        out['memory_gb'] = round(psutil.virtual_memory().total / 2**30)
        out['cores_physical'] = psutil.cpu_count(logical=False)
        out['cores_logical'] = psutil.cpu_count(logical=True)
    except ImportError:
        pass
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=20)
        if gpu.returncode == 0:
            out['gpus'] = [g.strip() for g in gpu.stdout.strip().splitlines()]
    except Exception:
        pass
    return out


def run(case, method, options=None, verbose=False, release=True):
    """Factorize case.A with the method and solve every right hand side. Returns the result dict."""
    options = dict(options or {})
    factorize = registry.get(method) if isinstance(method, str) else method
    name = method if isinstance(method, str) else getattr(method, '__name__', 'function')
    res = {'case': case.name, 'method': name, 'options': {k: repr(v) for k, v in options.items()},
           'n': case.n, 'nnz': int(case.A.nnz), 'when': dt.datetime.now().isoformat(timespec='seconds'),
           'machine': machine(), 'rhs': []}
    gc.collect()
    res['rss_before_gb'] = peak_rss_gb()
    t0 = time.perf_counter()
    try:
        solve = factorize(case.A, verbose, **options)
    except Exception as e:
        res.update(status='factorization failed', error=f'{type(e).__name__}: {e}',
                   traceback=traceback.format_exc(limit=5), factor_s=round(time.perf_counter() - t0, 3))
        return res
    res['factor_s'] = round(time.perf_counter() - t0, 3)
    owner = getattr(solve, '__self__', None)
    res['info'] = getattr(owner, 'info', '')
    status = 'ok'
    for k, b, xref in zip(case.meta['rhs_numbers'], case.bs, case.xs):
        t0 = time.perf_counter()
        try:
            x = np.asarray(solve(b))
        except Exception as e:
            res['rhs'].append({'rhs': k, 'error': f'{type(e).__name__}: {e}',
                               'solve_s': round(time.perf_counter() - t0, 3)})
            status = 'solve failed'
            continue
        seconds = time.perf_counter() - t0
        nb = max(np.abs(b).max(), 1e-300)
        relres = float(np.abs(b - case.A @ x).max() / nb)
        row = {'rhs': k, 'solve_s': round(seconds, 3), 'relres': relres}
        if xref is not None:
            row['err_ref'] = float(np.abs(x - xref).max() / max(np.abs(xref).max(), 1e-300))
        last = getattr(owner, 'last', None)
        if isinstance(last, tuple) and len(last) == 2:
            row['refinements'] = int(last[1])
        if not relres < ACCEPT:
            status = 'inaccurate'
        res['rhs'].append(row)
    res['peak_rss_gb'] = peak_rss_gb()
    solved = [r for r in res['rhs'] if 'solve_s' in r and 'error' not in r]
    res['solve_s_sum'] = round(sum(r['solve_s'] for r in solved), 3)
    res['total_s'] = round(res['factor_s'] + res['solve_s_sum'], 3)
    res['relres_max'] = max((r['relres'] for r in solved), default=None)
    res['status'] = status
    if release and owner is not None and callable(getattr(owner, 'release', None)):
        owner.release()
    return res


def save(res, folder=RESULTS):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = res['when'].replace(':', '').replace('-', '')
    method = res['method'].replace(':', '_').replace('/', '_')
    path = folder / f'{res["case"]}__{method}__{res["machine"]["host"]}__{stamp}.json'
    path.write_text(json.dumps(res, indent=1), encoding='utf-8')
    return path


def report(res):
    lines = [f'{res["case"]} with {res["method"]} {res["options"] or ""}: {res["status"]}']
    if 'error' in res:
        lines.append(f'  {res["error"]}')
        return '\n'.join(lines)
    lines.append(f'  factorization {res["factor_s"]:.2f} s  {res.get("info", "")}')
    for r in res['rhs']:
        if 'error' in r:
            lines.append(f'  b_{r["rhs"]:02d}: {r["error"]}')
            continue
        lines.append(f'  b_{r["rhs"]:02d}: solve {r["solve_s"]:7.3f} s  relres {r["relres"]:.1e}'
                     + (f'  vs reference {r["err_ref"]:.1e}' if 'err_ref' in r else '')
                     + (f'  refinements {r["refinements"]}' if 'refinements' in r else ''))
    lines.append(f'  total {res["total_s"]:.2f} s (factorization {res["factor_s"]:.2f} + solves '
                 f'{res["solve_s_sum"]:.2f}), peak memory {res["peak_rss_gb"]} GB')
    return '\n'.join(lines)


def parse_options(pairs):
    out = {}
    for p in pairs or []:
        if '=' not in p:
            raise SystemExit(f'--opt {p!r}: use name=value')
        k, v = p.split('=', 1)
        try:
            out[k.strip()] = ast.literal_eval(v.strip())
        except (ValueError, SyntaxError):
            out[k.strip()] = v.strip()
    return out


def compare(paths):
    """A table of result files (or folders of them)."""
    files = []
    for p in paths:
        p = Path(p)
        files += sorted(p.glob('*.json')) if p.is_dir() else [p]
    rows = []
    for f in files:
        r = json.loads(f.read_text(encoding='utf-8'))
        gpu = ', '.join(r['machine'].get('gpus', [])) or '-'
        rows.append((r['case'], r['method'], ' '.join(f'{k}={v}' for k, v in r['options'].items()), r['status'],
                     r.get('factor_s'), r.get('solve_s_sum'), r.get('total_s'), r.get('relres_max'),
                     r.get('peak_rss_gb'), r['machine'].get('processor') or r['machine']['platform'], gpu))
    head = ('case', 'method', 'options', 'status', 'factor s', 'solves s', 'total s', 'max relres', 'peak GB',
            'cpu', 'gpu')
    fmt = lambda v: '' if v is None else (f'{v:.2e}' if isinstance(v, float) and v < 1e-2 else
                                         f'{v:.2f}' if isinstance(v, float) else str(v))
    table = [head] + [tuple(fmt(v) for v in row) for row in sorted(rows, key=lambda r: (r[0], r[6] or 1e99))]
    widths = [max(len(row[i]) for row in table) for i in range(len(head))]
    return '\n'.join(' | '.join(c.ljust(w) for c, w in zip(row, widths)) for row in table)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m factorbench', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list', help='methods and downloaded cases')
    r = sub.add_parser('run', help='run a method on a case')
    r.add_argument('--case', required=True, nargs='+')
    r.add_argument('--method', required=True, nargs='+')
    r.add_argument('--opt', action='append', help='name=value (a python literal), repeatable')
    r.add_argument('--rhs', help='right hand sides to solve, e.g. 0,1,2 (default all)')
    r.add_argument('--data', default=str(cases.DATA))
    r.add_argument('--out', default=str(RESULTS))
    r.add_argument('--verbose', action='store_true')
    c = sub.add_parser('compare', help='table of result files')
    c.add_argument('paths', nargs='*', default=[str(RESULTS)])
    a = ap.parse_args(argv)

    if a.cmd == 'list':
        print('methods:')
        for k, v in registry.names().items():
            print(f'  {k:14s} {v}')
        print(f'cases in {cases.DATA}: {", ".join(cases.available()) or "none - run  python get_data.py"}')
        return
    if a.cmd == 'compare':
        print(compare(a.paths))
        return
    options = parse_options(a.opt)
    rhs = [int(k) for k in a.rhs.split(',')] if a.rhs else None
    for case_name in a.case:
        case = cases.load(case_name, a.data, rhs)
        print(case.describe(), flush=True)
        for method in a.method:
            res = run(case, method, options, verbose=a.verbose)
            print(report(res), flush=True)
            print(f'  -> {save(res, a.out)}', flush=True)
        del case
        gc.collect()

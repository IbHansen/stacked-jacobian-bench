# -*- coding: utf-8 -*-
"""
get_data.py - download the test cases from the GitHub release into data/ (python standard library only).

    python get_data.py                  makro100, the whole 100-year problem (150 MB download)
    python get_data.py --list

The cases and their sha256 checksums are in data_manifest.json.
"""
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT = ['makro100']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, target):
    tmp = target.with_suffix('.part')
    with urllib.request.urlopen(url) as r, open(tmp, 'wb') as f:
        total = int(r.headers.get('Content-Length') or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f'\r  {done / 2**20:6.0f} / {total / 2**20:.0f} MB', end='', flush=True)
    print()
    tmp.replace(target)


def get(name, manifest, data):
    case = manifest['cases'][name]
    folder = data / name
    if folder.exists() and all((folder / f).exists() for f in case['files']):
        print(f'{name}: already in {folder}')
        return
    url = f'https://github.com/{manifest["repo"]}/releases/download/{manifest["tag"]}/{case["file"]}'
    zpath = data / case['file']
    print(f'{name}: {url} ({case["bytes"] / 2**20:.0f} MB)')
    download(url, zpath)
    if sha256(zpath) != case['sha256']:
        zpath.unlink()
        sys.exit(f'{name}: checksum does not match data_manifest.json - download removed')
    expected = {f'{name}/{f}' for f in case['files']}
    with zipfile.ZipFile(zpath) as z:
        if set(z.namelist()) != expected:
            sys.exit(f'{name}: the zip does not hold the files data_manifest.json lists')
        folder.mkdir(parents=True, exist_ok=True)
        for member in expected:                   # only the listed names, written into the case folder
            with z.open(member) as src, open(folder / Path(member).name, 'wb') as dst:
                shutil.copyfileobj(src, dst)
    zpath.unlink()
    print(f'{name}: n={case["n"]:,}, nnz={case["nnz"]:,}, {case["rhs"]} right hand sides -> {folder}')


def main(args):
    manifest = json.loads((ROOT / 'data_manifest.json').read_text(encoding='utf-8'))
    if '--list' in args:
        for name, c in manifest['cases'].items():
            print(f'{name:10s} n={c["n"]:>10,}  nnz={c["nnz"]:>11,}  rhs={c["rhs"]}  {c["bytes"] / 2**20:5.0f} MB')
        return
    names = list(manifest['cases']) if args == ['all'] else (args or DEFAULT)
    unknown = set(names) - set(manifest['cases'])
    if unknown:
        sys.exit(f'unknown case(s) {sorted(unknown)}; use --list')
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    for name in names:
        get(name, manifest, data)


if __name__ == '__main__':
    main(sys.argv[1:])

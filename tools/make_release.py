# -*- coding: utf-8 -*-
"""
make_release.py - pack the cases in data/ as zip files for a GitHub release and write data_manifest.json.

    python tools/make_release.py                 all cases in data/
    python tools/make_release.py makro10 makro30

Then create the release (tag = TAG below) on GitHub and attach the files in release/, e.g. with the GitHub CLI:

    gh release create data-v1 release/*.zip --title "Test matrices v1" --notes-file release/NOTES.md

get_data.py downloads from  https://github.com/<repo>/releases/download/<tag>/<case>.zip  and checks the sha256
written here.
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = 'IbHansen/stacked-jacobian-bench'
TAG = 'data-v1'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main(names):
    data, out = ROOT / 'data', ROOT / 'release'
    out.mkdir(exist_ok=True)
    names = names or sorted(p.name for p in data.iterdir() if (p / 'A.npz').exists())
    manifest_path = ROOT / 'data_manifest.json'
    manifest = (json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists()
                else {'repo': REPO, 'tag': TAG, 'cases': {}})
    notes = ['Test matrices for factorbench (see README.md). One zip per case: J (A.npz), the right hand sides '
             '(b_NN.npy), the reference solutions (x_NN.npy), meta.json, names.json.', '']
    for name in names:
        folder = data / name
        zpath = out / f'{name}.zip'
        files = sorted(p for p in folder.iterdir() if p.suffix in ('.npz', '.npy', '.json'))
        with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in files:
                z.write(p, f'{name}/{p.name}')
        meta = json.loads((folder / 'meta.json').read_text(encoding='utf-8'))
        manifest['cases'][name] = {'file': zpath.name, 'bytes': zpath.stat().st_size, 'sha256': sha256(zpath),
                                   'n': meta['n'], 'nnz': meta['nnz'], 'rhs': len(meta['rhs']),
                                   'periods': meta['periods'], 'files': [p.name for p in files]}
        print(f'{zpath.name}: {zpath.stat().st_size / 2**20:.0f} MB, n={meta["n"]:,}, nnz={meta["nnz"]:,}')
    for name, c in sorted(manifest['cases'].items()):
        notes.append(f'- `{name}`: n = {c["n"]:,}, nnz = {c["nnz"]:,}, {c["rhs"]} right hand sides, periods '
                     f'{c["periods"][0]}-{c["periods"][1]}, {c["bytes"] / 2**20:.0f} MB')
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    (out / 'NOTES.md').write_text('\n'.join(notes) + '\n', encoding='utf-8')
    print(f'-> {manifest_path}, {out / "NOTES.md"}')


if __name__ == '__main__':
    main(sys.argv[1:])

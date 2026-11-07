"""Recompute paper analyses and verify them against the supplied references."""
import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal_value(a, b, tolerance):
    if a == b:
        return True
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)


def compare_table(actual, expected, tolerance):
    with actual.open(encoding='utf-8', newline='') as f:
        a = list(csv.reader(f, delimiter='\t'))
    with expected.open(encoding='utf-8', newline='') as f:
        b = list(csv.reader(f, delimiter='\t'))
    if len(a) != len(b) or not a or a[0] != b[0]:
        raise ValueError(f'Table schema/row mismatch: {actual.name}')
    for i, (ra, rb) in enumerate(zip(a[1:], b[1:]), 2):
        if len(ra) != len(rb):
            raise ValueError(f'Table column mismatch: {actual.name}, row {i}')
        for j, (va, vb) in enumerate(zip(ra, rb)):
            if not equal_value(va, vb, tolerance):
                raise ValueError(f'Value mismatch: {actual.name}, row {i}, column {a[0][j]}')
    return {'rows': len(a)-1, 'columns': len(a[0]), 'status': 'pass'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/paper_reproduction.json')
    parser.add_argument('--skip-figures', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text(encoding='utf-8'))
    out = ROOT / 'results/paper'
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    import numpy as np
    import matplotlib
    report = {'status': 'running', 'software': {'python': platform.python_version(),
              'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
              'scope': 'reanalysis of supplied source tables',
              'figures_requested': not args.skip_figures, 'tables': {}}
    try:
        checksums = json.loads((ROOT / 'data/paper/input_checksums.json').read_text())
        for name, expected in checksums.items():
            if digest(ROOT / name) != expected:
                raise ValueError(f'Input checksum mismatch: {name}')
        report['verified_inputs'] = len(checksums)
        with (out / 'selection.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, str(ROOT / cfg['selection_script']),
                            str(ROOT / cfg['selection_config'])], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        tolerance = cfg['numeric_tolerance']
        for name in cfg['expected_selection_tables']:
            report['tables'][name] = compare_table(ROOT / cfg['selection_output'] / name,
                                                   ROOT / 'reference/selection' / name, tolerance)
        if not args.skip_figures:
            with (out / 'figures.log').open('w', encoding='utf-8') as log:
                subprocess.run([sys.executable, str(ROOT / cfg['figure_script'])], cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            actual = json.loads((ROOT / cfg['figure_output'] / 'plotted_matrices.json').read_text())
            expected = json.loads((ROOT / 'reference/plotted_matrices.json').read_text())
            if len(actual) != len(expected):
                raise ValueError('Plotted matrix count differs')
            for a, b in zip(actual, expected):
                if a['title'] != b['title'] or a['shape'] != b['shape']:
                    raise ValueError('Plotted matrix metadata differs')
                av = [v for row in a['values'] for v in row]
                bv = [v for row in b['values'] for v in row]
                if len(av) != len(bv) or not all(equal_value(x, y, tolerance) for x, y in zip(av, bv)):
                    raise ValueError(f"Plotted matrix differs: {a['title']}")
            report['verified_plotted_matrices'] = len(actual)
            report['figure_files'] = [p.name for p in sorted((ROOT / cfg['figure_output']).glob('*.pdf'))]
        report['outputs'] = {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p)
                             for d in [ROOT / cfg['selection_output']] +
                             ([] if args.skip_figures else [ROOT / cfg['figure_output']])
                             for p in sorted(d.iterdir()) if p.is_file()}
        report['status'] = 'pass'
    except Exception as error:
        report['status'] = 'fail'
        report['error'] = str(error)
        raise
    finally:
        report['runtime_seconds'] = round(time.monotonic() - start, 3)
        (out / 'reproduction_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in report.items() if k not in ['outputs', 'tables']}, indent=2))


if __name__ == '__main__':
    main()

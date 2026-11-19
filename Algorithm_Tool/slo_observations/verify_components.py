"""Validate seven wrappers on an already exported SLO PNG; no NPZ or label access."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path('/opt/data/test/MMAgent')
SOURCE = Path('/opt/data/test/SLOctolyzer')
OUT = ROOT / 'docs/experiments/slo-tools-v1'
RUNNER = Path(__file__).with_name('runner.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'verification.json').exists():
        raise ValueError('verification_already_exists')
    image = SOURCE / 'outputs/fairseg_smoke/inputs/data_08001.png'
    commands = []
    files = []
    environment = {'PATH': str(Path(sys.executable).parent) + ':/usr/bin:/bin',
                   'PYTHONNOUSERSITE': '1', 'CUDA_VISIBLE_DEVICES': '', 'MPLBACKEND': 'Agg',
                   'TMPDIR': str(OUT), 'XDG_CACHE_HOME': str(OUT / 'cache'),
                   'MPLCONFIGDIR': str(OUT / 'cache/matplotlib'), 'OMP_NUM_THREADS': '8'}
    for task in ('vessels', 'avod', 'fovea', 'density', 'width', 'tortuosity', 'fractal'):
        output = OUT / (task + '.json')
        inp = image if task in ('vessels', 'avod', 'fovea') else OUT / 'vessel_mask.png'
        argv = [sys.executable, str(RUNNER), task, '--input', str(inp), '--output', str(output)]
        ports = {'vessels': ['vessel_mask'], 'avod': ['artery_mask', 'vein_mask', 'disc_mask'],
                 'fovea': ['fovea_map']}.get(task, [])
        for port in ports:
            path = OUT / (port + ('.npy' if port == 'fovea_map' else '.png'))
            argv += ['--' + port.replace('_', '-'), str(path)]
        result = subprocess.run(argv, cwd=ROOT, env=environment, capture_output=True,
                                text=True, timeout=180)
        (OUT / (task + '.log')).write_text(result.stdout + result.stderr)
        commands.append({'argv': argv, 'exit_code': result.returncode})
        if result.returncode:
            (OUT / 'failed-attempt.json').write_text(json.dumps(commands, indent=2))
            raise RuntimeError('component_failed_' + task)
        observation = json.loads(output.read_text())
        assert observation['input']['sha256'] == digest(inp)
        assert observation['clinical_effect'] is None and observation['research_only']
        if task not in ('vessels', 'avod', 'fovea'):
            assert observation['measurement']['value'] is not None
        files.append(output)
        for port in ports:
            path = OUT / (port + ('.npy' if port == 'fovea_map' else '.png'))
            assert observation['outputs'][port]['sha256'] == digest(path)
            files.append(path)
        print(task, 'passed', flush=True)
    # Match the installed official pipeline's original binary mask pixel-for-pixel.
    old = SOURCE / 'outputs/fairseg_smoke/data_08001/data_08001_slo_binary_map.png'
    a = np.asarray(Image.open(old))
    b = np.asarray(Image.open(OUT / 'vessel_mask.png'))
    assert a.shape == b.shape and np.array_equal(a, b), 'official_vessel_map_differs'
    unit = subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_runner'],
                          cwd=RUNNER.parent, env=environment, capture_output=True,
                          text=True, timeout=60)
    (OUT / 'unit.log').write_text(unit.stdout + unit.stderr)
    commands.append({'argv': [sys.executable, '-m', 'unittest', '-v', 'test_runner'],
                     'exit_code': unit.returncode})
    assert unit.returncode == 0
    files += [image, old, RUNNER, RUNNER.with_name('test_runner.py'), Path(__file__)]
    files += list(OUT.glob('*.log'))
    result = {'status': 'passed', 'tasks': ['vessels', 'avod', 'fovea', 'density', 'width',
               'tortuosity', 'fractal'], 'commands': commands, 'unit_tests': 16,
              'device': 'cpu', 'existing_export_only': True, 'labels_read': False,
              'official_vessel_map_equal': True, 'segmentation_accuracy_evaluated': False,
              'files': [{'path': str(p), 'sha256': digest(p)} for p in files]}
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()

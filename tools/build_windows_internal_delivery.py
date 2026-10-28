"""Build the Windows internal offline bundle from validated local artifacts."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import zipfile

from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename


ROOT = Path(__file__).resolve().parents[1]
NAME = 'unified-marl-0.7.0-windows-x64-cpu-py312-offline'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def build(destination):
    destination.mkdir(parents=True, exist_ok=True)
    folder = destination / NAME
    archive = destination / (NAME + '.zip')
    if folder.exists() or archive.exists():
        raise FileExistsError('Use a fresh output directory; existing delivery is preserved: ' + str(folder))
    folder.mkdir()
    wheels = folder / 'wheelhouse'
    wheels.mkdir()
    expected = {canonicalize_name(dist.metadata['Name']): dist.version for dist in metadata.distributions()
                if canonicalize_name(dist.metadata['Name']) not in {'pip', 'unified-marl'}}
    # This builder must run in the already validated full release environment.
    for name, version in {'torch': '2.6.0+cpu', 'numpy': '1.26.4', 'scipy': '1.14.1',
                          'scikit-learn': '1.5.2', 'gym': '0.26.2', 'pyzmq': '25.1.2',
                          'transformers': '4.53.0', 'ml-collections': '1.1.0'}.items():
        if expected.get(name) != version:
            raise RuntimeError('Use .venv-release to build: expected ' + name + '==' + version)
    acceptable_tags = set(sys_tags())
    wheel_index = {}
    for source in sorted((ROOT / '.work/wheelhouse').glob('*.whl')):
        name, version, _, tags = parse_wheel_filename(source.name)
        if tags & acceptable_tags:
            wheel_index.setdefault((canonicalize_name(name), str(version)), source)
    selected = []
    for name, version in sorted(expected.items()):
        source = wheel_index.get((name, version))
        if source is None:
            raise FileNotFoundError('Missing validated wheel: ' + name + '==' + version)
        target = wheels / source.name
        shutil.copyfile(source, target)
        selected.append((name, version, target))
    package = ROOT / 'dist/unified_marl-0.7.0-py3-none-any.whl'
    if digest(package) != 'ba6fec1d7eb805a86ef62ae61fa9e7a866bdbfc49ad6a1c8be1d3cbe27e445e7':
        raise ValueError('Original 0.7.0 wheel differs from the validated artifact')
    shutil.copyfile(package, wheels / package.name)
    selected.append(('unified-marl', '0.7.0', wheels / package.name))
    lines = ['# Windows x64 CPython 3.12 CPU; offline, exact validated dependencies.']
    lines += [f'{name}=={version} --hash=sha256:{digest(path)}' for name, version, path in selected]
    (folder/'requirements-lock.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    for source in (ROOT/'delivery/windows').iterdir():
        if source.is_file():
            shutil.copyfile(source, folder/source.name)
    for source in (ROOT/'delivery/common').glob('*.py'):
        shutil.copyfile(source, folder/source.name)
    for name in ['LICENSE', 'NOTICE']:
        shutil.copyfile(ROOT/name, folder/name)
    for name in ['licenses', 'examples']:
        for source in (ROOT/name).rglob('*'):
            if source.is_file() and source.suffix in {'.py', '.yaml', '.md', '.txt'} and '__pycache__' not in source.parts:
                target = folder/source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    (folder/'docs').mkdir()
    for source in (ROOT/'docs').iterdir():
        if source.is_file() and source.suffix in {'.md', '.csv'}:
            shutil.copyfile(source, folder/'docs'/source.name)
    (folder/'source').mkdir()
    shutil.copyfile(ROOT/'dist/unified_marl-0.7.0.tar.gz', folder/'source/unified_marl-0.7.0.tar.gz')
    (folder/'data').mkdir()
    (folder/'data/README.md').write_text('真实数据不随包提供。通过 --dataset 指定数据路径。可用 examples/make_demo_data.py 生成 toy 合成演示数据；它不是真实实验数据。\n', encoding='utf-8')
    files = [{'path': path.relative_to(folder).as_posix(), 'size': path.stat().st_size, 'sha256': digest(path)}
             for path in sorted(folder.rglob('*')) if path.is_file()]
    manifest = {'version': '0.7.0', 'delivery_revision': 1, 'edition': 'windows-x64-cpu-py312',
                'python_runtime_bundled': False, 'python': 'CPython 3.12 x64',
                'includes': ['core', 'role', 'hmf', 'reward', 'distributed'],
                'excludes': ['real datasets', 'checkpoints', 'external simulators', 'CUDA', 'virtual environments'],
                'files': files}
    (folder/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                output.write(path, path.relative_to(destination).as_posix(),
                             compress_type=zipfile.ZIP_STORED if path.suffix in {'.whl', '.gz'} else zipfile.ZIP_DEFLATED)
    result = {'edition': manifest['edition'], 'folder': str(folder), 'archive': str(archive),
              'size': archive.stat().st_size, 'sha256': digest(archive), 'files': len(files), 'wheels': len(selected)}
    (destination/(NAME+'.build.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'dist/internal')
    build(parser.parse_args().output.resolve())

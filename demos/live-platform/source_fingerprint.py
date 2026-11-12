"""Hash the local Docker COPY inputs, without requiring Python on the host.

Keep INPUTS and excluded() aligned with Dockerfile and Dockerfile.dockerignore.
The Docker image stores its fingerprint before pip can generate build files.
Launchers run --check inside the existing image with a read-only source mount.
"""
import argparse
import hashlib
import json
from pathlib import Path

INPUTS = ('pyproject.toml', 'README.md', 'LICENSE', 'NOTICE', 'licenses', 'src', 'demos/live-platform')
VENDOR = ('demos', 'live-platform', 'bundle', 'vendor', 'CityFlow')


def excluded(relative):
    parts = relative.parts
    if '__pycache__' in parts or relative.suffix == '.pyc':
        return True
    if parts[:3] == ('demos', 'live-platform', 'local-state'):
        return True
    if parts[:5] == VENDOR and len(parts) > 5:
        return parts[5] in ('.git', 'build') or parts[5].endswith('.egg-info')
    return False


def fingerprint(root):
    root = Path(root).resolve()
    paths = []
    for name in INPUTS:
        path = root / name
        if not path.exists():
            raise ValueError(f'Missing Docker build input: {name}')
        paths.extend([path] if path.is_file() else (p for p in path.rglob('*') if p.is_file()))
    digest = hashlib.sha256()
    count = 0
    for path in sorted(paths, key=lambda p: p.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        if excluded(relative):
            continue
        digest.update(relative.as_posix().encode('utf-8') + b'\0')
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        count += 1
    return dict(schema_version=1, sha256=digest.hexdigest(), files=count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    if bool(args.write) == bool(args.check):
        parser.error('Choose exactly one of --write or --check')
    try:
        if args.write:
            result = fingerprint(args.write)
            (args.write / '.local-source-fingerprint.json').write_text(json.dumps(result), encoding='utf-8')
            print(json.dumps(result))
            return 0
        receipt = Path('/opt/unified-marl/.local-source-fingerprint.json')
        if not receipt.exists():
            print('The existing image predates automatic source checking; rebuilding once.')
            return 1
        expected = json.loads(receipt.read_text(encoding='utf-8'))
        actual = fingerprint(args.check)
        if expected != actual:
            print('Local source changed; the image needs to be rebuilt.')
            return 1
        print('Local image matches this source download; no build or network needed.')
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f'Could not verify local build inputs: {exc}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

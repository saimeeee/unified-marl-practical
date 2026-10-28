"""Verify internal archives and target dependency metadata without installing them."""
from email.parser import BytesParser
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]
EXTRAS = ('', 'role', 'hmf', 'reward', 'distributed')


def verify(archive):
    linux = '-linux-' in archive.name
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
        assert len(names) == len(set(names)) == len({name.casefold() for name in names})
        assert all(not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts for name in names)
        prefix = names[0].split('/')[0]+'/'
        assert all(name.startswith(prefix) for name in names)
        manifest = json.loads(package.read(prefix+'manifest.json'))
        network = {}
        if prefix+'requirements-network.txt' in names:
            for line in package.read(prefix+'requirements-network.txt').decode('utf-8').splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    requirement = Requirement(line)
                    network[canonicalize_name(requirement.name)] = requirement
        expected = {prefix+item['path']:item for item in manifest['files']}
        assert len(expected) == len(manifest['files'])
        assert set(expected)|{prefix+'manifest.json'} == set(names)
        wheels = {}
        for name, entry in expected.items():
            content = package.read(name)
            assert len(content) == entry['size'] and hashlib.sha256(content).hexdigest() == entry['sha256'], name
            if name.endswith('.whl'):
                parsed_name, version, _, tags = parse_wheel_filename(PurePosixPath(name).name)
                if linux:
                    platforms = ['linux_x86_64', 'manylinux2014_x86_64', 'manylinux2010_x86_64', 'manylinux1_x86_64']
                    platforms += ['manylinux_2_'+str(minor)+'_x86_64' for minor in range(5,29)]
                else:
                    platforms = ['win_amd64']
                allowed = set(cpython_tags((3,12), abis=['cp312'], platforms=platforms))
                allowed.update(compatible_tags((3,12), interpreter='cp312', platforms=platforms))
                assert tags & allowed, 'Wrong target wheel: '+name
                with zipfile.ZipFile(io.BytesIO(content)) as wheel:
                    metadata_names = [item for item in wheel.namelist()
                                      if len(PurePosixPath(item).parts)==2 and item.endswith('.dist-info/METADATA')]
                    assert len(metadata_names)==1, name
                    metadata = BytesParser().parsebytes(wheel.read(metadata_names[0]))
                assert canonicalize_name(metadata['Name']) == canonicalize_name(parsed_name)
                assert Version(metadata['Version']) == version
                assert '3.12.14' in SpecifierSet(metadata.get('Requires-Python',''))
                normalized = canonicalize_name(parsed_name)
                assert normalized not in wheels
                wheels[normalized] = (version, metadata)
        env = {'implementation_name':'cpython','implementation_version':'3.12.14',
               'os_name':'posix' if linux else 'nt','platform_machine':'x86_64' if linux else 'AMD64',
               'platform_python_implementation':'CPython','platform_release':'',
               'platform_system':'Linux' if linux else 'Windows','platform_version':'',
               'python_full_version':'3.12.14','python_version':'3.12',
               'sys_platform':'linux' if linux else 'win32'}
        checked, deferred = 0, []
        for name, (version, metadata) in wheels.items():
            for raw in metadata.get_all('Requires-Dist', []):
                requirement = Requirement(raw)
                extras = EXTRAS if name == 'unified-marl' else ('',)
                needed = any(requirement.marker is None or requirement.marker.evaluate(dict(env, extra=extra)) for extra in extras)
                if not needed:
                    continue
                dependency = canonicalize_name(requirement.name)
                if dependency not in wheels and linux and dependency == 'hf-xet':
                    assert dependency in network, 'Missing explicit network requirement: '+raw
                    assert str(network[dependency].specifier) == str(requirement.specifier)
                    deferred.append(raw)
                    continue
                assert dependency in wheels, name+' missing '+raw
                assert wheels[dependency][0] in requirement.specifier, name+' conflicting '+raw
                checked += 1
        assert wheels['torch'][0] == Version('2.6.0+cpu')
        return {'archive':archive.name, 'bytes':archive.stat().st_size,
                'sha256':hashlib.file_digest(archive.open('rb'),'sha256').hexdigest(),
                'status':'PASS', 'manifest_files':len(expected), 'wheels':len(wheels),
                'requirements_checked':checked, 'target':'linux-x86_64' if linux else 'windows-x64',
                'network_at_install':deferred,
                'linux_runtime_verified':False, 'scope':'archive hashes, wheel tags and target metadata only'}


if __name__ == '__main__':
    reports = [verify(path) for path in sorted((ROOT/'dist/internal').glob('unified-marl-0.7.0-*.zip'))]
    assert {report['target'] for report in reports} == {'windows-x64','linux-x86_64'}
    output = ROOT/'dist/internal/archive-verification.json'
    output.write_text(json.dumps(reports, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(reports, ensure_ascii=True))

"""Reconstruct the accepted install ZIPs on GitHub and publish a private release.

This only downloads wheels and reconstructs files; it never installs the Linux
package, invokes its scripts, runs training or fetches the deferred hf-xet wheel.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TAG = 'v0.7.0-internal.1'
REPOSITORY = 'saimeeee/unified-marl'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def gh(*args, json_result=True):
    result = subprocess.run(['gh', *args], cwd=ROOT, check=False, capture_output=True, text=True, encoding='utf-8')
    if result.returncode:
        raise RuntimeError('GitHub command failed: '+result.stderr.strip())
    return json.loads(result.stdout) if json_result and result.stdout.strip() else result.stdout


def prepare(download=True, cache=None):
    blueprint = json.loads((ROOT/'delivery/release/blueprint.json').read_text(encoding='utf-8'))
    output = ROOT/'release-output'
    output.mkdir(exist_ok=False)
    reports = []
    for bundle in blueprint['bundles']:
        wheels = (cache/bundle['edition']) if cache else ROOT/'.release-cache'/bundle['edition']
        wheels.mkdir(parents=True, exist_ok=True)
        if download:
            command = [sys.executable, '-m', 'pip', 'download', '--disable-pip-version-check',
                       '--no-deps', '--no-cache-dir', '--only-binary=:all:', '--require-hashes',
                       '--python-version', '312', '--implementation', 'cp',
                       '--abi', 'cp312', '--abi', 'abi3', '--abi', 'none',
                       '--index-url', 'https://pypi.org/simple',
                       '--extra-index-url', 'https://download.pytorch.org/whl/cpu',
                       '--dest', str(wheels), '-r', str(ROOT/'delivery/release'/('requirements-'+bundle['edition']+'.txt'))]
            platforms = ['win_amd64'] if bundle['edition']=='windows' else ['manylinux_2_28_x86_64','manylinux_2_17_x86_64','manylinux2014_x86_64','linux_x86_64']
            for platform in platforms:
                command += ['--platform', platform]
            subprocess.run(command, check=True, cwd=ROOT)
        by_hash = {sha(path):path for path in wheels.glob('*.whl')}
        archive_path = output/bundle['archive']
        with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for item in bundle['entries']:
                if 'source' in item:
                    source = (ROOT/item['source']).resolve()
                    if not source.is_relative_to(ROOT):
                        raise ValueError('Unsafe input path')
                else:
                    source = by_hash[item['sha256']]
                if source.stat().st_size != item['size'] or sha(source) != item['sha256']:
                    raise ValueError('Input hash mismatch: '+item['filename'])
                info = zipfile.ZipInfo(item['filename'], tuple(item['date_time']))
                for key, value in item.items():
                    if key not in {'filename','date_time','source','download','sha256','size','comment','extra'}:
                        setattr(info, key, value)
                info.comment = bytes.fromhex(item['comment'])
                info.extra = bytes.fromhex(item['extra'])
                archive.writestr(info, source.read_bytes(), compresslevel=6)
        actual = sha(archive_path)
        if actual != bundle['sha256'] or archive_path.stat().st_size != bundle['size']:
            raise ValueError('Reconstructed ZIP differs from accepted artifact: '+bundle['archive'])
        reports.append({'name':archive_path.name,'sha256':actual,'size':archive_path.stat().st_size})
    import shutil
    for source in (ROOT/'delivery/release/assets').iterdir():
        shutil.copyfile(source, output/source.name)
    for name in ['unified_marl-0.7.0-py3-none-any.whl','unified_marl-0.7.0.tar.gz']:
        shutil.copyfile(ROOT/'delivery/release/vendor'/name, output/name)
    expected = json.loads((ROOT/'delivery/release/asset-manifest.json').read_text(encoding='utf-8'))
    for item in expected:
        path = output/item['name']
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Final release asset mismatch: '+path.name)
    print(json.dumps({'prepared':len(expected),'zip_reports':reports,'linux_tests_run':False}))
    return output, expected


def publish(output, expected):
    if os.environ.get('GITHUB_REPOSITORY') != REPOSITORY:
        raise RuntimeError('Refusing to publish from a different repository')
    repo = gh('api', 'repos/'+REPOSITORY)
    if not repo.get('private'):
        raise RuntimeError('Repository must remain private')
    commit = os.environ['GITHUB_SHA']
    releases = gh('api','repos/'+REPOSITORY+'/releases?per_page=100')
    existing = next((r for r in releases if r['tag_name']==TAG), None)
    if existing and not existing['draft']:
        raise RuntimeError('Published tag already exists; refusing to replace it')
    if existing:
        previous = existing['target_commitish']
        if previous != commit:
            comparison = gh('api','repos/'+REPOSITORY+'/compare/'+previous+'...'+commit)
            if comparison['status'] not in {'ahead','identical'}:
                raise RuntimeError('Existing draft is unrelated to this commit')
        notes = (ROOT/'docs/releases/v0.7.0-internal.1.md').read_text(encoding='utf-8')
        if (existing.get('body') or '').replace('\r\n','\n').strip() != notes.replace('\r\n','\n').strip():
            raise RuntimeError('Existing draft notes differ; refusing replacement')
    else:
        gh('release','create',TAG,'--repo',REPOSITORY,'--target',commit,'--draft',
           '--title','unified-marl 0.7.0 — 内部 Windows / Linux 安装包',
           '--notes-file',str(ROOT/'docs/releases/v0.7.0-internal.1.md'),json_result=False)
    drafts = gh('api','repos/'+REPOSITORY+'/releases?per_page=100')
    remote = next(r for r in drafts if r['tag_name']==TAG)
    for item in expected:
        found = next((a for a in remote['assets'] if a['name']==item['name']), None)
        if found:
            if found['size'] != item['bytes'] or found.get('digest') != 'sha256:'+item['sha256']:
                raise RuntimeError('Different existing asset; refusing overwrite: '+item['name'])
        else:
            gh('release','upload',TAG,str(output/item['name']),'--repo',REPOSITORY,json_result=False)
    remote = gh('api','repos/'+REPOSITORY+'/releases/'+str(remote['id']))
    for item in expected:
        found = next(a for a in remote['assets'] if a['name']==item['name'])
        if found['size'] != item['bytes'] or found.get('digest') != 'sha256:'+item['sha256']:
            raise RuntimeError('Uploaded asset verification failed: '+item['name'])
    gh('api','--method','PATCH','repos/'+REPOSITORY+'/releases/'+str(remote['id']),
       '-f','target_commitish='+commit,'-F','draft=false','-f','make_latest=true')
    print(json.dumps({'status':'PUBLISHED','url':remote['html_url'],'assets':len(expected),'private':True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--local-cache', type=Path)
    args = parser.parse_args()
    output, expected = prepare(download=args.local_cache is None, cache=args.local_cache)
    if args.publish:
        publish(output, expected)

#!/usr/bin/env python3
"""Build the pinned Linux x86_64 CPU/Python 3.12 mixed-dependency delivery ZIP."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]
EDITION = "linux-x86_64-cpu-py312-network-completion"
VERSION = Version("0.7.0")
BUNDLE_NAME = f"unified-marl-{VERSION}-linux-x86_64-cpu-py312"
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
PYPI_INDEX = "https://pypi.org/simple"
TARGET_PLATFORMS = (
    "manylinux_2_28_x86_64",
    "manylinux_2_17_x86_64",
    "manylinux2014_x86_64",
    "linux_x86_64",
)
TARGET_ABIS = ("cp312", "abi3", "none")
ROOT_EXTRAS = ("role", "hmf", "reward", "distributed")
EXPECTED_CORE_SHA256 = "ba6fec1d7eb805a86ef62ae61fa9e7a866bdbfc49ad6a1c8be1d3cbe27e445e7"
EXPECTED_SDIST_SHA256 = "c4319d229cb702b0fe9c8010158a6275497b8a74d77f8ce1d10570cd05aeea7e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wheel_info(path: Path):
    name, version, _build, tags = parse_wheel_filename(path.name)
    return canonicalize_name(name), version, tags


def tag_is_supported(tag) -> bool:
    if tag.platform == "any":
        platform_ok = True
    elif tag.platform == "linux_x86_64":
        platform_ok = True
    elif tag.platform == "manylinux2014_x86_64":
        platform_ok = True
    else:
        match = re.fullmatch(r"manylinux_(\d+)_(\d+)_x86_64", tag.platform)
        platform_ok = bool(match and tuple(map(int, match.groups())) <= (2, 28))
    if not platform_ok:
        return False
    if tag.interpreter in {"py3", "py2.py3"} and tag.abi == "none":
        return True
    if tag.interpreter == "cp312" and tag.abi in {"cp312", "none"}:
        return True
    if tag.abi == "abi3" and re.fullmatch(r"cp3\d+", tag.interpreter):
        return int(tag.interpreter[3:]) <= 12
    return False


def wheel_metadata(path: Path):
    with zipfile.ZipFile(path) as archive:
        entries = [
            name for name in archive.namelist()
            if len(PurePosixPath(name).parts) == 2
            and PurePosixPath(name).parts[0].endswith(".dist-info")
            and PurePosixPath(name).name == "METADATA"
        ]
        if len(entries) != 1:
            raise RuntimeError(f"Expected one wheel METADATA in {path.name}, found {len(entries)}")
        return BytesParser().parsebytes(archive.read(entries[0]))


def installed_versions(release_python: Path) -> dict[str, str]:
    code = (
        "import importlib.metadata as m,json; "
        "print(json.dumps({d.metadata['Name']:d.version for d in m.distributions() "
        "if d.metadata.get('Name') and d.metadata['Name'].lower() not in ('pip','unified-marl')}))"
    )
    result = subprocess.run(
        [str(release_python), "-X", "utf8", "-I", "-c", code],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    raw = json.loads(result.stdout)
    return {canonicalize_name(name): version for name, version in raw.items()}


def release_python_path() -> Path:
    candidates = (ROOT / ".venv-release" / "Scripts" / "python.exe", ROOT / ".venv-release" / "bin" / "python")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(".venv-release Python was not found; use the validated project environment.")


def ensure_core_wheel() -> Path:
    wheel = ROOT / "dist" / "unified_marl-0.7.0-py3-none-any.whl"
    if not wheel.is_file():
        raise RuntimeError(f"Core wheel is missing: {wheel}")
    actual = sha256(wheel)
    if actual != EXPECTED_CORE_SHA256:
        raise RuntimeError(f"Core wheel SHA-256 changed: expected {EXPECTED_CORE_SHA256}, got {actual}")
    name, version, tags = wheel_info(wheel)
    if name != "unified-marl" or version != VERSION or not any(tag_is_supported(tag) for tag in tags):
        raise RuntimeError("Core wheel name, version or universal wheel tags are unexpected.")
    return wheel


def ensure_sdist() -> Path:
    source = ROOT / "dist" / "unified_marl-0.7.0.tar.gz"
    if not source.is_file():
        raise RuntimeError(f"Source distribution is missing: {source}")
    actual = sha256(source)
    if actual != EXPECTED_SDIST_SHA256:
        raise RuntimeError(f"Source distribution SHA-256 changed: expected {EXPECTED_SDIST_SHA256}, got {actual}")
    return source


def marker_environment() -> dict[str, str]:
    return {
        "implementation_name": "cpython",
        "implementation_version": "3.12.14",
        "os_name": "posix",
        "platform_machine": "x86_64",
        "platform_python_implementation": "CPython",
        "platform_release": "",
        "platform_system": "Linux",
        "platform_version": "",
        "python_full_version": "3.12.14",
        "python_version": "3.12",
        "sys_platform": "linux",
        "extra": "",
    }


def pip_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PIP_CONFIG_FILE"] = os.devnull
    env["PIP_NO_INPUT"] = "1"
    env["PIP_EXTRA_INDEX_URL"] = ""
    env["PIP_KEYRING_PROVIDER"] = "disabled"
    return env


def download_wheel(release_python: Path, wheelhouse: Path, name: str, version: str) -> None:
    cached = ROOT / ".work" / "wheelhouse"
    wanted_name = canonicalize_name(name)
    wanted_version = Version(version)
    for candidate in cached.glob("*.whl"):
        cached_name, cached_version, cached_tags = wheel_info(candidate)
        if (
            cached_name == wanted_name
            and cached_version == wanted_version
            and any(tag.platform == "any" and tag.abi == "none" and tag.interpreter in {"py3", "py2.py3"} for tag in cached_tags)
        ):
            destination = wheelhouse / candidate.name
            shutil.copy2(candidate, destination)
            print(f"Reused cached universal wheel: {candidate.name}", flush=True)
            return
    command = [
        str(release_python), "-X", "utf8", "-I", "-m", "pip", "download",
        "--disable-pip-version-check", "--no-input", "--dest", str(wheelhouse),
        "--no-deps", "--only-binary=:all:", "--python-version", "312",
        "--implementation", "cp",
    ]
    for abi in TARGET_ABIS:
        command.extend(("--abi", abi))
    for target in TARGET_PLATFORMS:
        command.extend(("--platform", target))
    if canonicalize_name(name) == "torch":
        command.extend(("--index-url", TORCH_INDEX))
    else:
        command.extend(("--index-url", PYPI_INDEX))
    command.append(f"{name}=={version}")
    print(f"Downloading Linux wheel: {name}=={version}", flush=True)
    subprocess.run(command, check=True, cwd=ROOT, env=pip_env())


def select_wheel(wheelhouse: Path, name: str, version: str) -> Path | None:
    wanted_name = canonicalize_name(name)
    wanted_version = Version(version)
    candidates = []
    for path in wheelhouse.glob("*.whl"):
        parsed_name, parsed_version, tags = wheel_info(path)
        if parsed_name == wanted_name and parsed_version == wanted_version and any(tag_is_supported(tag) for tag in tags):
            candidates.append(path)
    return sorted(candidates, key=lambda path: path.name.lower())[0] if candidates else None


def collect_closure(core_wheel: Path, release_python: Path, wheelhouse: Path) -> tuple[dict[str, str], dict[str, str]]:
    versions = installed_versions(release_python)
    env = marker_environment()
    selected: dict[str, str] = {}
    requested_extras: dict[str, set[str]] = {}
    pending: list[str] = []
    processed: set[tuple[str, str]] = set()
    network_requirements: dict[str, set[str]] = {}

    def add_requirement(raw: str, active_extra: str) -> None:
        requirement = Requirement(raw)
        marker_env = dict(env, extra=active_extra)
        if requirement.marker is not None and not requirement.marker.evaluate(marker_env):
            return
        if requirement.url:
            raise RuntimeError(f"Direct URL dependency is not allowed in the offline closure: {raw}")
        name = canonicalize_name(requirement.name)
        if name == "unified-marl":
            if requirement.specifier and not requirement.specifier.contains(VERSION, prereleases=True):
                raise RuntimeError(f"Core package version conflicts with dependency: {raw}")
            return
        pin = versions.get(name)
        if pin is None:
            if name == "hf-xet":
                network_requirements.setdefault(name, set()).add(str(requirement.specifier))
                return
            raise RuntimeError(f"Validated .venv-release does not contain a required package: {requirement.name}")
        resolved_version = Version(pin)
        if requirement.specifier and not requirement.specifier.contains(resolved_version, prereleases=True):
            raise RuntimeError(f".venv-release pin {requirement.name}=={pin} does not satisfy {raw}")
        previous = selected.get(name)
        if previous is not None and Version(previous) != resolved_version:
            raise RuntimeError(f"Conflicting pins for {requirement.name}: {previous} and {pin}")
        is_new = name not in selected
        selected[name] = pin
        extras = requested_extras.setdefault(name, set())
        extra_count = len(extras)
        extras.update(canonicalize_name(item) for item in requirement.extras)
        if is_new or len(extras) != extra_count:
            pending.append(name)

    core_meta = wheel_metadata(core_wheel)
    if canonicalize_name(core_meta.get("Name", "")) != "unified-marl" or Version(core_meta.get("Version", "0")) != VERSION:
        raise RuntimeError("Core wheel metadata does not match unified-marl 0.7.0.")
    for raw in core_meta.get_all("Requires-Dist", []):
        add_requirement(raw, "")
        for extra in ROOT_EXTRAS:
            add_requirement(raw, extra)

    while pending:
        name = pending.pop(0)
        selected_version = selected[name]
        wheel = select_wheel(wheelhouse, name, selected_version)
        if wheel is None:
            download_wheel(release_python, wheelhouse, name, selected_version)
            wheel = select_wheel(wheelhouse, name, selected_version)
        if wheel is None:
            raise RuntimeError(f"No CPython 3.12 Linux x86_64 binary wheel found for {name}=={selected_version}.")
        actual_name, actual_version, tags = wheel_info(wheel)
        if actual_name != name or actual_version != Version(selected_version) or not any(tag_is_supported(tag) for tag in tags):
            raise RuntimeError(f"Downloaded wheel is not a supported Linux target: {wheel.name}")
        metadata = wheel_metadata(wheel)
        if canonicalize_name(metadata.get("Name", "")) != name or Version(metadata.get("Version", "0")) != actual_version:
            raise RuntimeError(f"Wheel filename/metadata mismatch: {wheel.name}")
        extras_to_check = {""} | requested_extras.get(name, set())
        for raw in metadata.get_all("Requires-Dist", []):
            for extra in extras_to_check:
                key = (name, extra)
                if key not in processed:
                    add_requirement(raw, extra)
        processed.update((name, extra) for extra in extras_to_check)

    # Re-read actual Linux wheel metadata and prove that every active dependency is in the closure.
    wheel_metadata_by_name = {}
    for name, version in selected.items():
        wheel = select_wheel(wheelhouse, name, version)
        if wheel is None:
            raise RuntimeError(f"Selected wheel disappeared from the wheelhouse: {name}=={version}")
        wheel_metadata_by_name[name] = wheel_metadata(wheel)
    for name, metadata in wheel_metadata_by_name.items():
        for raw in metadata.get_all("Requires-Dist", []):
            requirement = Requirement(raw)
            active_extras = {""} | requested_extras.get(name, set())
            if requirement.marker is not None and not any(
                requirement.marker.evaluate(dict(env, extra=extra)) for extra in active_extras
            ):
                continue
            dependency = canonicalize_name(requirement.name)
            if dependency == "unified-marl":
                continue
            if dependency in network_requirements:
                if str(requirement.specifier) not in network_requirements[dependency]:
                    raise RuntimeError(f"Network complement constraint differs from Linux wheel metadata: {name} -> {raw}")
                continue
            if dependency not in selected:
                raise RuntimeError(f"Linux wheel metadata dependency missing from closure: {name} -> {raw}")
            dependency_version = Version(selected[dependency])
            if requirement.specifier and not requirement.specifier.contains(dependency_version, prereleases=True):
                raise RuntimeError(f"Linux wheel metadata pin mismatch: {name} requires {raw}, selected {dependency_version}")
    if set(network_requirements) != {"hf-xet"} or network_requirements["hf-xet"] != {"<2.0.0,>=1.1.3"}:
        raise RuntimeError(f"Unexpected Linux-only network dependency requirements: {network_requirements}")
    return dict(sorted(selected.items())), {"hf-xet": "hf-xet>=1.1.3,<2.0.0"}


def copy_file(source: Path, target_root: Path, relative: str) -> None:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or "\\" in relative or ":" in relative:
        raise RuntimeError(f"Unsafe packaged path: {relative}")
    destination = target_root.joinpath(*posix.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def safe_remove_tree(path: Path, expected_parent: Path) -> None:
    parent = expected_parent.resolve(strict=True)
    if path.is_symlink():
        raise RuntimeError(f"Refusing to recursively remove a symlink: {path}")
    target = path.resolve(strict=True)
    if target == parent or not target.is_relative_to(parent):
        raise RuntimeError(f"Refusing recursive removal outside the owned directory: {target}")
    if not target.is_dir():
        raise RuntimeError(f"Refusing recursive removal of a non-directory: {target}")
    shutil.rmtree(target)


def create_requirements(target_root: Path, selected: dict[str, str], core_wheel: Path) -> None:
    hashes_by_name: dict[str, list[str]] = {"unified-marl": [sha256(core_wheel)]}
    for wheel in sorted((target_root / "wheelhouse").glob("*.whl"), key=lambda path: path.name.lower()):
        name, version, tags = wheel_info(wheel)
        if not any(tag_is_supported(tag) for tag in tags) or any(tag.platform.startswith(("win", "macosx")) for tag in tags):
            raise RuntimeError(f"Non-Linux or incompatible wheel reached the delivery wheelhouse: {wheel.name}")
        if name == "unified-marl":
            if version != VERSION:
                raise RuntimeError(f"Unexpected unified-marl wheel in wheelhouse: {wheel.name}")
            continue
        if name not in selected or version != Version(selected[name]):
            raise RuntimeError(f"Unselected wheel reached the delivery wheelhouse: {wheel.name}")
        hashes_by_name.setdefault(name, []).append(sha256(wheel))
    lines = [
        "# CPython 3.12 / Linux x86_64 / CPU; bundled wheel subset with exact versions and SHA-256 hashes.",
        "# The installer uses --no-deps for this local subset; requirements-network.txt adds Linux-only hf-xet.",
    ]
    for name in ["unified-marl", *selected.keys()]:
        hashes = hashes_by_name.get(name)
        if not hashes:
            raise RuntimeError(f"No wheel hash available for requirements entry {name}.")
        if name == "unified-marl":
            spec = f"unified-marl[{','.join(ROOT_EXTRAS)}]=={VERSION}"
        else:
            spec = f"{name}=={selected[name]}"
        lines.append(spec + " " + " ".join(f"--hash=sha256:{value}" for value in sorted(set(hashes))))
    (target_root / "requirements-local-linux-cpu.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def create_network_requirements(target_root: Path, requirements: dict[str, str]) -> None:
    if requirements != {"hf-xet": "hf-xet>=1.1.3,<2.0.0"}:
        raise RuntimeError(f"Unexpected network-complement set: {requirements}")
    lines = [
        "# Required by huggingface-hub 0.36.2 Linux x86_64 wheel metadata.",
        "# This one Linux-only dependency is resolved from PyPI by the recipient; it is not bundled or hash-locked.",
        requirements["hf-xet"],
    ]
    (target_root / "requirements-network.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def build_delivery(core_wheel: Path, sdist: Path, selected: dict[str, str], network_requirements: dict[str, str], work_root: Path) -> tuple[Path, int]:
    dist_internal = ROOT / "dist" / "internal"
    dist_internal.mkdir(parents=True, exist_ok=True)
    dist_internal_resolved = dist_internal.resolve(strict=True)
    if not dist_internal_resolved.is_relative_to((ROOT / "dist").resolve(strict=True)):
        raise RuntimeError(f"dist/internal resolves outside the project dist directory: {dist_internal_resolved}")
    bundle_dir = dist_internal / BUNDLE_NAME
    zip_path = dist_internal / f"{BUNDLE_NAME}.zip"
    if bundle_dir.exists() or zip_path.exists():
        raise RuntimeError(f"Linux delivery output already exists; inspect before replacing: {bundle_dir} / {zip_path}")

    temporary = Path(tempfile.mkdtemp(prefix="build-", dir=work_root))
    temporary_resolved = temporary.resolve(strict=True)
    if not temporary_resolved.is_relative_to(work_root.resolve(strict=True)):
        raise RuntimeError(f"Build temporary directory resolves outside .work/internal-linux: {temporary_resolved}")
    stage = temporary / BUNDLE_NAME
    stage.mkdir()
    successful = False
    try:
        local_files = {
            core_wheel: f"wheelhouse/{core_wheel.name}",
            sdist: "source/unified_marl-0.7.0.tar.gz",
            ROOT / "LICENSE": "LICENSE",
            ROOT / "NOTICE": "NOTICE",
            ROOT / "docs" / "SUPPORT_MATRIX.md": "docs/SUPPORT_MATRIX.md",
            ROOT / "docs" / "KNOWN_ISSUES.md": "docs/KNOWN_ISSUES.md",
            ROOT / "docs" / "LICENSES.md": "docs/LICENSES.md",
            ROOT / "docs" / "INTEGRATION_SPEC.md": "docs/INTEGRATION_SPEC.md",
            ROOT / "docs" / "ZMQ_PROTOCOL.md": "docs/ZMQ_PROTOCOL.md",
            ROOT / "docs" / "INTERNAL_DISTRIBUTION.md": "docs/INTERNAL_DISTRIBUTION.md",
            ROOT / "src" / "unified_marl" / "PROVENANCE.json": "PROVENANCE.json",
            ROOT / "delivery" / "linux" / "README.md": "README.md",
            ROOT / "delivery" / "linux" / "examples-README.md": "examples/README.md",
            ROOT / "delivery" / "linux" / "install.sh": "install.sh",
            ROOT / "delivery" / "linux" / "run.sh": "run.sh",
            ROOT / "delivery" / "linux" / "data-README.md": "data/README.md",
            ROOT / "delivery" / "common" / "verify_bundle.py": "verify_bundle.py",
            ROOT / "delivery" / "common" / "smoke_test.py": "smoke_test.py",
        }
        for source, relative in local_files.items():
            if not source.is_file():
                raise RuntimeError(f"Required delivery input is missing: {source}")
            copy_file(source, stage, relative)
        for license_file in sorted((ROOT / "licenses").glob("*.txt")):
            copy_file(license_file, stage, f"licenses/{license_file.name}")
        for example in sorted((ROOT / "examples").iterdir(), key=lambda path: path.name.lower()):
            if example.is_file() and example.name != "README.md" and example.suffix.lower() in {".py", ".yaml"}:
                copy_file(example, stage, f"examples/{example.name}")
        source_wheelhouse = work_root / "wheelhouse"
        for wheel in sorted(source_wheelhouse.glob("*.whl"), key=lambda path: path.name.lower()):
            name, version, tags = wheel_info(wheel)
            if name not in selected or version != Version(selected[name]) or not any(tag_is_supported(tag) for tag in tags) or any(tag.platform.startswith(("win", "macosx")) for tag in tags):
                raise RuntimeError(f"Refusing to package unselected or non-Linux wheel: {wheel.name}")
            copy_file(wheel, stage, f"wheelhouse/{wheel.name}")
        create_requirements(stage, selected, core_wheel)
        create_network_requirements(stage, network_requirements)

        files = []
        for path in sorted(stage.rglob("*"), key=lambda item: item.relative_to(stage).as_posix().lower()):
            if not path.is_file():
                continue
            relative = path.relative_to(stage).as_posix()
            files.append({"path": relative, "size": path.stat().st_size, "sha256": sha256(path)})
        manifest = {
            "edition": EDITION,
            "package": "unified-marl",
            "version": str(VERSION),
            "target": "CPython 3.12 / Linux x86_64 / CPU; glibc >= 2.28",
            "files": files,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

        temporary_zip = temporary / f"{BUNDLE_NAME}.zip"
        if temporary_zip.exists():
            raise RuntimeError(f"Temporary ZIP path already exists; inspect before replacing: {temporary_zip}")
        with zipfile.ZipFile(temporary_zip, "x", compression=zipfile.ZIP_STORED) as archive:
            for path in sorted(stage.rglob("*"), key=lambda item: item.relative_to(stage).as_posix().lower()):
                if path.is_file():
                    relative = path.relative_to(stage).as_posix()
                    archive.write(path, f"{BUNDLE_NAME}/{relative}", compress_type=zipfile.ZIP_STORED)
        if bundle_dir.exists() or zip_path.exists():
            raise RuntimeError("Linux delivery output appeared during the build; refusing to overwrite it.")
        shutil.copytree(stage, bundle_dir)
        os.replace(temporary_zip, zip_path)
        successful = True
        return zip_path, len(files)
    finally:
        if successful:
            safe_remove_tree(temporary, work_root)


def main() -> int:
    if platform.system() != "Windows":
        print("This build target downloads Linux wheels but must run from the validated project environment.")
    work_root = ROOT / ".work" / "internal-linux"
    wheelhouse = work_root / "wheelhouse"
    work_root.mkdir(parents=True, exist_ok=True)
    wheelhouse.mkdir(parents=True, exist_ok=True)
    core_wheel = ensure_core_wheel()
    sdist = ensure_sdist()
    release_python = release_python_path()
    selected, network_requirements = collect_closure(core_wheel, release_python, wheelhouse)
    zip_path, file_count = build_delivery(core_wheel, sdist, selected, network_requirements, work_root)
    print(json.dumps({
        "status": "BUILT",
        "edition": EDITION,
        "zip": str(zip_path.resolve()),
        "size": zip_path.stat().st_size,
        "sha256": sha256(zip_path),
        "manifest_files": file_count,
        "dependency_wheels": len(selected),
        "dependencies": selected,
        "network_complements": network_requirements,
        "linux_runtime_test": "not run; package preparation only, per user instruction",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        print(f"Linux mixed delivery build failed: {error}", file=sys.stderr)
        raise SystemExit(1)

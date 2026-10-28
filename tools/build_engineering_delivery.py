"""Build the 0.8.0 engineering bundles from the validated 0.7.0 wheelhouses.

This utility is intentionally separate from the existing internal delivery
builders. It builds the universal project wheel locally, reuses checked-in
internal wheelhouse artifacts, and never downloads dependencies.
"""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import Tag
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]
VERSION = Version("0.8.0")
WHEEL_NAME = f"unified_marl-{VERSION}-py3-none-any.whl"
WINDOWS_SOURCE = ROOT / "dist/internal/unified-marl-0.7.0-windows-x64-cpu-py312-offline"
LINUX_SOURCE = ROOT / "dist/internal/unified-marl-0.7.0-linux-x86_64-cpu-py312"
WINDOWS_STEM = f"unified-marl-{VERSION}-windows-x64-cpu-py312-engineering"
LINUX_STEM = f"unified-marl-{VERSION}-linux-x86_64-cpu-py312-engineering"
ROOT_PACKAGE = "unified-marl"
EXPECTED_SOURCE_VERSION = Version("0.7.0")
FEATURE_EXTRAS = ("role", "hmf", "reward", "distributed")
REQUIREMENT_HASH = re.compile(r"--hash=sha256:([0-9a-fA-F]{64})")
MANYLINUX = re.compile(r"manylinux_(\d+)_(\d+)_x86_64\Z")

REQUIRED_DOC_FILES = (
    "APPLICATIONS_GUIDE.md",
    "WAREHOUSE_GUIDE.md",
    "TRAFFIC_GUIDE.md",
    "POWERGRID_GUIDE.md",
    "ENGINEERING_DISTRIBUTION.md",
)
EXAMPLE_SUFFIXES = {
    ".py", ".yaml", ".yml", ".md", ".json", ".csv", ".txt", ".toml",
    ".ini", ".xml", ".svg", ".png", ".jpg", ".jpeg", ".gif",
}
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", "venv"}
OPTIONAL_REQUIREMENT_FILES = (
    "requirements-applications.txt",
    "requirements-powergrid.txt",
    "requirements-traffic.txt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wheel_metadata(path: Path):
    with zipfile.ZipFile(path) as archive:
        metadata_names = [
            name for name in archive.namelist()
            if len(PurePosixPath(name).parts) == 2
            and PurePosixPath(name).parts[0].endswith(".dist-info")
            and PurePosixPath(name).name == "METADATA"
        ]
        if len(metadata_names) != 1:
            raise RuntimeError(f"Expected one METADATA file in {path.name}, found {len(metadata_names)}")
        return BytesParser().parsebytes(archive.read(metadata_names[0]))


def wheel_entry_points(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        names = [
            name for name in archive.namelist()
            if len(PurePosixPath(name).parts) == 2
            and PurePosixPath(name).parts[0].endswith(".dist-info")
            and PurePosixPath(name).name == "entry_points.txt"
        ]
        return archive.read(names[0]).decode("utf-8") if names else ""


def checked_manifest(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Validated source bundle manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files", [])
    if not entries:
        raise RuntimeError(f"Source bundle manifest has no entries: {manifest_path}")
    seen: set[str] = set()
    for entry in entries:
        name = entry["path"]
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise RuntimeError(f"Unsafe path in source bundle manifest: {name}")
        key = name.casefold()
        if key in seen:
            raise RuntimeError(f"Duplicate path in source bundle manifest: {name}")
        seen.add(key)
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise RuntimeError(f"Source bundle manifest path escapes its root: {name}")
        if not path.is_file():
            raise RuntimeError(f"Source bundle is missing manifest entry: {name}")
        if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Source bundle integrity check failed: {name}")
    return manifest


def load_lock(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        requirement_text, separator, hash_text = line.partition(" --hash=")
        if not separator:
            raise RuntimeError(f"Lock entry has no SHA-256 hash at {path}:{line_number}")
        requirement = Requirement(requirement_text)
        exact = [(item.operator, item.version) for item in requirement.specifier]
        if len(exact) != 1 or exact[0][0] != "==":
            raise RuntimeError(f"Lock entry is not pinned to one version at {path}:{line_number}")
        hashes = {value.lower() for value in REQUIREMENT_HASH.findall("--hash=" + hash_text)}
        if not hashes:
            raise RuntimeError(f"Lock entry has no valid SHA-256 hash at {path}:{line_number}")
        name = canonicalize_name(requirement.name)
        if name in records:
            raise RuntimeError(f"Duplicate package in lock file: {requirement.name}")
        records[name] = {
            "name": requirement.name,
            "canonical_name": name,
            "version": Version(exact[0][1]),
            "hashes": hashes,
            "extras": set(requirement.extras),
        }
    return records


def wheel_target_ok(tags: set[Tag], target: str) -> bool:
    for tag in tags:
        if tag.platform == "any":
            platform_ok = True
        elif target == "windows":
            platform_ok = tag.platform == "win_amd64"
        elif tag.platform in {"linux_x86_64", "manylinux2014_x86_64"}:
            platform_ok = target == "linux"
        elif target == "linux":
            match = MANYLINUX.fullmatch(tag.platform)
            platform_ok = bool(match and tuple(map(int, match.groups())) <= (2, 28))
        else:
            platform_ok = False
        if not platform_ok:
            continue
        if tag.interpreter in {"py3", "py2.py3"} and tag.abi == "none":
            return True
        if tag.interpreter == "cp312" and tag.abi in {"cp312", "none"}:
            return True
        if tag.abi == "abi3" and re.fullmatch(r"cp3\d+", tag.interpreter):
            if int(tag.interpreter[3:]) <= 12:
                return True
    return False


def index_source_wheels(source: Path, target: str) -> dict[tuple[str, Version], list[Path]]:
    wheelhouse = source / "wheelhouse"
    if not wheelhouse.is_dir():
        raise FileNotFoundError(f"Validated source wheelhouse is missing: {wheelhouse}")
    result: dict[tuple[str, Version], list[Path]] = {}
    for path in sorted(wheelhouse.glob("*.whl"), key=lambda item: item.name.casefold()):
        name, version, _build, tags = parse_wheel_filename(path.name)
        if wheel_target_ok(tags, target):
            result.setdefault((canonicalize_name(name), version), []).append(path)
    return result


def selected_wheels(source: Path, records: dict[str, dict], target: str) -> tuple[dict[str, dict], dict[str, Path]]:
    index = index_source_wheels(source, target)
    selected: dict[str, dict] = {}
    paths_by_hash: dict[str, Path] = {}
    root_record = records.get(ROOT_PACKAGE)
    if not root_record or root_record["version"] != EXPECTED_SOURCE_VERSION:
        raise RuntimeError(f"Expected a {EXPECTED_SOURCE_VERSION} root package lock entry in {source}")
    for name, record in records.items():
        key = (name, record["version"])
        candidates = index.get(key, [])
        matching = [candidate for candidate in candidates if sha256(candidate) in record["hashes"]]
        if not matching:
            raise FileNotFoundError(f"No validated {target} wheel matches lock entry {name}=={record['version']}")
        selected[name] = {**record, "paths": matching}
        for candidate in matching:
            paths_by_hash[sha256(candidate)] = candidate
    return selected, paths_by_hash


def release_python() -> Path:
    candidate = ROOT / ".venv-release/Scripts/python.exe"
    if not candidate.is_file():
        raise FileNotFoundError(f"Validated build interpreter not found: {candidate}")
    code = (
        "import platform,struct,sys; "
        "assert platform.python_implementation()=='CPython' and sys.version_info[:2]==(3,12); "
        "assert sys.platform=='win32' and struct.calcsize('P')==8; "
        "print(sys.executable)"
    )
    subprocess.run([str(candidate), "-I", "-c", code], cwd=ROOT, check=True)
    return candidate.resolve()


def build_universal_wheel(interpreter: Path, work: Path) -> Path:
    wheel_dir = work / "built-wheel"
    wheel_dir.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update({
        "PIP_NO_INDEX": "1",
        "PIP_NO_INPUT": "1",
        "PIP_NO_CACHE_DIR": "1",
        "PIP_CONFIG_FILE": "NUL",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUTF8": "1",
    })
    command = [
        str(interpreter), "-I", "-m", "pip", "wheel", "--disable-pip-version-check",
        "--no-deps", "--no-build-isolation", "--wheel-dir", str(wheel_dir), str(ROOT),
    ]
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    candidates = list(wheel_dir.glob("*.whl"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one local project wheel, found {len(candidates)}")
    wheel = candidates[0]
    name, version, _build, tags = parse_wheel_filename(wheel.name)
    if canonicalize_name(name) != ROOT_PACKAGE or version != VERSION or Tag("py3", "none", "any") not in tags:
        raise RuntimeError(f"Unexpected project wheel built from source: {wheel.name}")
    metadata = wheel_metadata(wheel)
    if metadata.get("Name", "").casefold().replace("_", "-") != ROOT_PACKAGE or Version(metadata["Version"]) != VERSION:
        raise RuntimeError("Built wheel METADATA does not identify unified-marl 0.8.0")
    if not re.search(r"(?m)^unified-marl-app\s*=", wheel_entry_points(wheel)):
        raise RuntimeError("Built wheel has no unified-marl-app console command")
    return wheel


def marker_environment(target: str) -> dict[str, str]:
    env = default_environment()
    env.update({
        "implementation_name": "cpython",
        "implementation_version": "3.12.0",
        "platform_python_implementation": "CPython",
        "python_full_version": "3.12.0",
        "python_version": "3.12",
        "platform_machine": "AMD64" if target == "windows" else "x86_64",
        "platform_system": "Windows" if target == "windows" else "Linux",
        "platform_release": "",
        "platform_version": "",
        "sys_platform": "win32" if target == "windows" else "linux",
        "os_name": "nt" if target == "windows" else "posix",
    })
    return env


def active_for_extras(requirement: Requirement, environment: dict[str, str], extras: set[str]) -> bool:
    if requirement.marker is None:
        return True
    for extra in {"", *extras}:
        values = dict(environment)
        values["extra"] = extra
        if requirement.marker.evaluate(values):
            return True
    return False


def resolve_offline_lock(
    project_wheel: Path,
    extras: set[str],
    available: dict[str, dict],
    target: str,
) -> tuple[dict[str, dict], dict[str, Requirement]]:
    environment = marker_environment(target)
    selected: dict[str, dict] = {}
    missing_network: dict[str, Requirement] = {}
    queue: list[tuple[str, set[str], Path | None]] = [(ROOT_PACKAGE, extras, project_wheel)]
    visited: set[tuple[str, tuple[str, ...]]] = set()
    while queue:
        name, package_extras, wheel = queue.pop(0)
        state = (name, tuple(sorted(package_extras)))
        if state in visited:
            continue
        visited.add(state)
        if name == ROOT_PACKAGE:
            current_wheel = project_wheel
        else:
            record = available.get(name)
            if record is None:
                continue
            current_wheel = record["paths"][0]
            selected[name] = record
        metadata = wheel_metadata(current_wheel)
        for raw in metadata.get_all("Requires-Dist", []):
            requirement = Requirement(raw)
            dependency = canonicalize_name(requirement.name)
            if not active_for_extras(requirement, environment, package_extras):
                continue
            if dependency == ROOT_PACKAGE:
                continue
            record = available.get(dependency)
            if record is None:
                if target == "linux" and dependency == "hf-xet":
                    missing_network[dependency] = requirement
                    continue
                raise RuntimeError(
                    f"{target} dependency {requirement} is absent from the validated local wheelhouse; "
                    "add an explicit feature-specific network requirement before building."
                )
            version = record["version"]
            if requirement.specifier and not requirement.specifier.contains(version, prereleases=True):
                raise RuntimeError(f"Pinned {dependency}=={version} does not satisfy {requirement}")
            previous = selected.get(dependency)
            if previous and previous["version"] != version:
                raise RuntimeError(f"Conflicting local versions selected for {dependency}")
            selected[dependency] = record
            queue.append((dependency, set(requirement.extras), None))
    selected[ROOT_PACKAGE] = {
        "name": ROOT_PACKAGE,
        "canonical_name": ROOT_PACKAGE,
        "version": VERSION,
        "paths": [project_wheel],
        "extras": set(extras),
    }
    return selected, missing_network


def lock_lines(selected: dict[str, dict], extras: set[str] | None = None) -> list[str]:
    lines = []
    for name in sorted(selected):
        record = selected[name]
        if name == ROOT_PACKAGE and extras:
            spec = f"{ROOT_PACKAGE}[{','.join(sorted(extras))}]=={record['version']}"
        else:
            spec = f"{name}=={record['version']}"
        hashes = sorted({sha256(path) for path in record["paths"]})
        if not hashes:
            raise RuntimeError(f"No wheel hashes available for {spec}")
        lines.append(spec + " " + " ".join(f"--hash=sha256:{value}" for value in hashes))
    return lines


def copy_file(source: Path, stage: Path, relative: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required engineering delivery input is missing: {source}")
    target = stage / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def copy_examples(stage: Path) -> None:
    examples = ROOT / "examples"
    if not examples.is_dir():
        raise FileNotFoundError(f"Examples directory is missing: {examples}")
    for source in sorted(examples.rglob("*"), key=lambda item: item.as_posix().casefold()):
        relative = source.relative_to(examples)
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        if not source.is_file() or source.is_symlink() or source.suffix.lower() not in EXAMPLE_SUFFIXES:
            continue
        copy_file(source, stage, (Path("examples") / relative).as_posix())


def copy_requirements(stage: Path) -> None:
    for name in OPTIONAL_REQUIREMENT_FILES:
        source = ROOT / name
        if source.is_file():
            copy_file(source, stage, name)
    local_powergrid = ROOT / "delivery/engineering/requirements-powergrid.txt"
    if local_powergrid.is_file() and not (stage / "requirements-powergrid.txt").exists():
        copy_file(local_powergrid, stage, "requirements-powergrid.txt")


def copy_documentation(stage: Path) -> None:
    docs = ROOT / "docs"
    for name in REQUIRED_DOC_FILES:
        if not (docs / name).is_file():
            raise FileNotFoundError(f"Required engineering guide is missing: {docs / name}")
    for source in sorted(docs.iterdir(), key=lambda item: item.name.casefold()):
        if source.is_file() and not source.is_symlink() and source.suffix.lower() in {".md", ".csv"}:
            copy_file(source, stage, (Path("docs") / source.name).as_posix())


def create_platform_stage(
    stage: Path,
    target: str,
    source_bundle: Path,
    project_wheel: Path,
) -> dict:
    manifest = checked_manifest(source_bundle)
    if Version(str(manifest.get("version"))) != EXPECTED_SOURCE_VERSION:
        raise RuntimeError(f"Unexpected source bundle version in {source_bundle}")
    lock_name = "requirements-lock.txt" if target == "windows" else "requirements-local-linux-cpu.txt"
    old_records = load_lock(source_bundle / lock_name)
    available, _ = selected_wheels(source_bundle, old_records, target)
    old_root = old_records[ROOT_PACKAGE]
    if old_root["version"] != EXPECTED_SOURCE_VERSION:
        raise RuntimeError(f"Unexpected old core package lock in {source_bundle}")

    wheelhouse = stage / "wheelhouse"
    wheelhouse.mkdir(parents=True)
    new_project_copy = wheelhouse / WHEEL_NAME
    shutil.copyfile(project_wheel, new_project_copy)
    copied: dict[tuple[str, Version, str], Path] = {}
    for name, record in available.items():
        if name == ROOT_PACKAGE:
            continue
        for source in record["paths"]:
            target_path = wheelhouse / source.name
            shutil.copyfile(source, target_path)
            copied[(name, record["version"], source.name)] = target_path
        record["paths"] = [wheelhouse / source.name for source in record["paths"]]
    if target == "windows":
        # Confirm the refreshed wheel still fits the complete validated local
        # closure for the package extras that this delivery supports.
        for extras in [set(), *({feature} for feature in FEATURE_EXTRAS), set(FEATURE_EXTRAS)]:
            resolve_offline_lock(project_wheel, extras, available, target)
        new_records = {name: record for name, record in available.items() if name != ROOT_PACKAGE}
        new_records[ROOT_PACKAGE] = {
            "name": ROOT_PACKAGE, "canonical_name": ROOT_PACKAGE, "version": VERSION,
            "paths": [new_project_copy], "extras": set(),
        }
        (stage / "requirements-lock.txt").write_text(
            "# Windows x64 CPython 3.12 CPU; exact hashes from the validated local wheelhouse.\n"
            + "\n".join(lock_lines(new_records)) + "\n",
            encoding="utf-8", newline="\n",
        )
    else:
        feature_sets = {"core": set(), **{feature: {feature} for feature in FEATURE_EXTRAS}}
        for feature, extras in feature_sets.items():
            selected, network = resolve_offline_lock(
                new_project_copy, extras, available, target,
            )
            suffix = "core" if feature == "core" else feature
            requirement_file = stage / f"requirements-{suffix}-linux.txt"
            requirement_file.write_text(
                "# Linux x86_64 / CPython 3.12 / CPU; install this feature's exact local wheels.\n"
                + "\n".join(lock_lines(selected, extras)) + "\n",
                encoding="utf-8", newline="\n",
            )
            if network:
                missing = sorted(network)
                if missing != ["hf-xet"]:
                    raise RuntimeError(f"Unexpected Linux network complements for {feature}: {missing}")
                requirements = sorted({str(item) for item in network.values()})
                (stage / f"requirements-network-{feature}.txt").write_text(
                    "# Recipient-side network complement for this optional feature; not bundled.\n"
                    + "\n".join(requirements) + "\n",
                    encoding="utf-8", newline="\n",
                )

    copy_file(ROOT / "LICENSE", stage, "LICENSE")
    copy_file(ROOT / "NOTICE", stage, "NOTICE")
    for license_file in sorted((ROOT / "licenses").rglob("*"), key=lambda item: item.as_posix().casefold()):
        if license_file.is_file() and not license_file.is_symlink():
            copy_file(license_file, stage, (Path("licenses") / license_file.relative_to(ROOT / "licenses")).as_posix())
    copy_file(ROOT / "README.md", stage, "README.md")
    copy_documentation(stage)
    for relative in (
        "windows/install.ps1", "windows/run.ps1", "linux/install.sh", "linux/run.sh",
    ):
        copy_file(ROOT / "delivery/engineering" / relative, stage, relative)
    copy_file(ROOT / "delivery/common/verify_bundle.py", stage, "verify_bundle.py")
    copy_examples(stage)
    copy_requirements(stage)

    content_files = [
        path for path in sorted(stage.rglob("*"), key=lambda item: item.relative_to(stage).as_posix().casefold())
        if path.is_file()
    ]
    files = [
        {"path": path.relative_to(stage).as_posix(), "size": path.stat().st_size, "sha256": sha256(path)}
        for path in content_files
    ]
    included_wheels = [path.name for path in sorted(wheelhouse.glob("*.whl"), key=lambda item: item.name.casefold())]
    bundle_manifest = {
        "edition": "windows-x64-cpu-py312-engineering" if target == "windows" else "linux-x86_64-cpu-py312-engineering",
        "package": ROOT_PACKAGE,
        "version": str(VERSION),
        "target": "CPython 3.12 x64 / Windows / CPU" if target == "windows" else "CPython 3.12 / Linux x86_64 / CPU / glibc >= 2.28",
        "python_runtime_bundled": False,
        "application_dependencies": "feature-specific; see docs/APPLICATIONS_GUIDE.md and docs/ENGINEERING_DISTRIBUTION.md",
        "bundled_wheels": included_wheels,
        "files": files,
    }
    (stage / "manifest.json").write_text(
        json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n",
    )
    return bundle_manifest


def create_zip(stage: Path, temporary_zip: Path, root_name: str) -> None:
    with zipfile.ZipFile(temporary_zip, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(stage.rglob("*"), key=lambda item: item.relative_to(stage).as_posix().casefold()):
            if path.is_file():
                relative = path.relative_to(stage).as_posix()
                compression = zipfile.ZIP_STORED if path.suffix.lower() == ".whl" else zipfile.ZIP_DEFLATED
                archive.write(path, f"{root_name}/{relative}", compress_type=compression)


def build(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    final_wheel = ROOT / "dist" / WHEEL_NAME
    final_outputs = [
        (WINDOWS_STEM, output / WINDOWS_STEM, output / f"{WINDOWS_STEM}.zip"),
        (LINUX_STEM, output / LINUX_STEM, output / f"{LINUX_STEM}.zip"),
    ]
    sidecars = [output / f"{stem}.zip.sha256" for stem, _, _ in final_outputs]
    report_path = output / "engineering-delivery-build.json"
    existing = [
        str(path)
        for path in [final_wheel, *(item for row in final_outputs for item in row[1:]), *sidecars, report_path]
        if path.exists()
    ]
    if existing:
        raise FileExistsError("Engineering delivery output already exists; inspect before rebuilding: " + ", ".join(existing))
    if not (ROOT / "dist").is_dir():
        raise FileNotFoundError("Project dist directory is missing")
    if not (ROOT / ".work").is_dir():
        raise FileNotFoundError("Project .work directory is missing; create it before running this builder")

    work_root = ROOT / ".work" / "engineering-delivery"
    resolved_workspace = ROOT.resolve()
    resolved_work_root = work_root.resolve()
    if resolved_work_root == resolved_workspace or not resolved_work_root.is_relative_to(resolved_workspace):
        raise RuntimeError(f"Refusing to build outside the workspace: {resolved_work_root}")
    work_root.mkdir(parents=True, exist_ok=True)
    resolved_work_root = work_root.resolve()
    if resolved_work_root == resolved_workspace or not resolved_work_root.is_relative_to(resolved_workspace):
        raise RuntimeError(f"Refusing to build outside the workspace: {resolved_work_root}")
    temporary = Path(tempfile.mkdtemp(prefix="build-", dir=work_root))
    try:
        interpreter = release_python()
        project_wheel = build_universal_wheel(interpreter, temporary)
        stages = []
        for target, source_bundle, stem in (
            ("windows", WINDOWS_SOURCE, WINDOWS_STEM),
            ("linux", LINUX_SOURCE, LINUX_STEM),
        ):
            stage = temporary / stem
            stage.mkdir()
            create_platform_stage(stage, target, source_bundle, project_wheel)
            zip_path = temporary / f"{stem}.zip"
            create_zip(stage, zip_path, stem)
            stages.append((stem, stage, zip_path, target))

        # Build both archives fully before publishing any new output.
        if final_wheel.exists() or any(directory.exists() or archive.exists() for _, directory, archive in final_outputs) \
                or any(path.exists() for path in sidecars) or report_path.exists():
            raise FileExistsError("Engineering delivery output appeared during the build; refusing to overwrite it")
        shutil.copyfile(project_wheel, final_wheel)
        results = []
        for stem, stage, temporary_zip, target in stages:
            directory = output / stem
            archive = output / f"{stem}.zip"
            shutil.copytree(stage, directory)
            os.replace(temporary_zip, archive)
            archive_hash = sha256(archive)
            (output / f"{stem}.zip.sha256").write_text(
                f"{archive_hash}  {archive.name}\n", encoding="ascii", newline="\n",
            )
            results.append({
                "target": target,
                "directory": str(directory),
                "archive": str(archive),
                "size": archive.stat().st_size,
                "sha256": archive_hash,
                "files": len(json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["files"]),
            })
        result = {
            "status": "BUILT",
            "version": str(VERSION),
            "wheel": str(final_wheel),
            "wheel_sha256": sha256(final_wheel),
            "bundles": results,
            "build_mode": "local wheelhouse reuse; no dependency downloads or training",
        }
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n",
        )
        return result
    finally:
        resolved_workspace = ROOT.resolve()
        resolved_work_root = work_root.resolve()
        resolved_temporary = temporary.resolve()
        if not resolved_work_root.is_relative_to(resolved_workspace):
            raise RuntimeError(f"Refusing temporary cleanup outside the workspace: {resolved_work_root}")
        if resolved_temporary == resolved_work_root or not resolved_temporary.is_relative_to(resolved_work_root):
            raise RuntimeError(f"Refusing temporary cleanup outside the engineering build directory: {resolved_temporary}")
        shutil.rmtree(temporary, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/internal")
    args = parser.parse_args()
    try:
        result = build(args.output.resolve())
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        print(f"Engineering delivery build failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

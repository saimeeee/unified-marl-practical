"""Prepare 0.8.0 engineering release assets and a reproducible ZIP blueprint.

This tool consumes already-built local artifacts. It does not download wheels,
install packages, publish a release, run applications, or start training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.8.0"
WINDOWS_ARCHIVE = f"unified-marl-{VERSION}-windows-x64-cpu-py312-engineering.zip"
LINUX_ARCHIVE = f"unified-marl-{VERSION}-linux-x86_64-cpu-py312-engineering.zip"
WHEEL_NAME = f"unified_marl-{VERSION}-py3-none-any.whl"
SDIST_NAME = f"unified_marl-{VERSION}.tar.gz"
APPLICATION_GUIDES = (
    "docs/APPLICATIONS_GUIDE.md",
    "docs/WAREHOUSE_GUIDE.md",
    "docs/TRAFFIC_GUIDE.md",
    "docs/POWERGRID_GUIDE.md",
    "docs/ENGINEERING_DISTRIBUTION.md",
)
APPLICATION_REQUIREMENTS = (
    "requirements-applications.txt",
    "requirements-powergrid.txt",
    "requirements-traffic.txt",
)
REQUIREMENTS_FILES = {
    "windows": "delivery/release/requirements-windows.txt",
    "linux": "delivery/release/requirements-linux.txt",
}
REQUIREMENT_HASH = re.compile(r"--hash=sha256:([0-9a-fA-F]{64})")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_stream(stream) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def json_write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def ensure_project_file(path: Path, description: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"{description} must stay inside the project workspace: {path}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {description}: {resolved}")
    return resolved


def safe_member_parts(name: str) -> tuple[str, ...]:
    if not name or "\\" in name or "\x00" in name:
        raise ValueError(f"Unsafe ZIP member name: {name!r}")
    member = PurePosixPath(name)
    if member.is_absolute() or any(part in {"", ".", ".."} for part in member.parts):
        raise ValueError(f"Unsafe ZIP member path: {name!r}")
    if any(":" in part for part in member.parts):
        raise ValueError(f"Drive-like component in ZIP member path: {name!r}")
    return member.parts


def old_wheel_index(blueprint_path: Path) -> dict[str, dict]:
    blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))
    if not isinstance(blueprint.get("bundles"), list):
        raise ValueError("The previous release blueprint has no bundles list")
    index: dict[str, dict] = {}
    for bundle in blueprint["bundles"]:
        for entry in bundle.get("entries", []):
            if not str(entry.get("filename", "")).lower().endswith(".whl"):
                continue
            digest = entry.get("sha256")
            size = entry.get("size")
            if not isinstance(digest, str) or len(digest) != 64 or not isinstance(size, int):
                raise ValueError("Malformed old wheel entry in release blueprint")
            record = index.setdefault(digest, {"size": size, "sources": [], "downloads": []})
            if record["size"] != size:
                raise ValueError(f"The old blueprint assigns conflicting sizes to wheel SHA-256 {digest}")
            source = entry.get("source")
            download = entry.get("download")
            if isinstance(source, str) and source not in record["sources"]:
                record["sources"].append(source)
            if isinstance(download, str) and download not in record["downloads"]:
                record["downloads"].append(download)
    return index


def old_source_for(digest: str, size: int, old_index: dict[str, dict]) -> str | None:
    mapping = old_index.get(digest)
    if mapping is None:
        raise ValueError(f"No old release SHA mapping exists for dependency wheel {digest}; refusing to invent a source")
    if mapping["size"] != size:
        raise ValueError(f"Old release size mapping differs for dependency wheel {digest}")
    for source_text in mapping["sources"]:
        source_parts = safe_member_parts(source_text)
        source = (ROOT / Path(*source_parts)).resolve()
        if not source.is_relative_to(ROOT.resolve()):
            raise ValueError(f"Unsafe old release source path: {source_text}")
        if source.is_file() and source.stat().st_size == size and sha256_file(source) == digest:
            return source_text
    if mapping["downloads"]:
        # The new publisher resolves source-less entries through this old
        # SHA-indexed blueprint and performs any permitted download later.
        return None
    raise ValueError(f"Old release SHA mapping has neither a verified local source nor a download for {digest}")


def verify_zip_sidecar(path: Path, digest: str) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Built bundle SHA-256 sidecar is missing: {sidecar}")
    fields = sidecar.read_text(encoding="ascii").strip().split()
    if len(fields) < 1 or fields[0].lower() != digest:
        raise ValueError(f"Built bundle SHA-256 sidecar differs from the archive: {sidecar}")
    if len(fields) >= 2 and fields[-1] != path.name:
        raise ValueError(f"Built bundle SHA-256 sidecar names a different file: {sidecar}")


def entry_blueprint(info: zipfile.ZipInfo, digest: str, size: int, source: str | None) -> dict:
    result = {
        "filename": info.filename,
        "date_time": list(info.date_time),
        "compress_type": info.compress_type,
        "create_system": info.create_system,
        "create_version": info.create_version,
        "extract_version": info.extract_version,
        "flag_bits": info.flag_bits,
        "internal_attr": info.internal_attr,
        "external_attr": info.external_attr,
        "reserved": info.reserved,
        "volume": info.volume,
        "comment": info.comment.hex(),
        "extra": info.extra.hex(),
        "sha256": digest,
        "size": size,
    }
    if source is not None:
        result["source"] = source
    return result


def stream_member_to_file(archive: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path) -> tuple[str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with archive.open(info, "r") as source, target.open("xb") as destination:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
            destination.write(block)
    return digest.hexdigest(), size


def hash_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[str, int]:
    with archive.open(info, "r") as stream:
        return sha256_stream(stream)


def validate_embedded_manifest(archive: zipfile.ZipFile, infos: list[zipfile.ZipInfo], root_name: str,
                               actual_records: dict[str, dict]) -> None:
    manifest_name = f"{root_name}/manifest.json"
    matches = [item for item in infos if item.filename == manifest_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one embedded manifest at {manifest_name}")
    manifest = json.loads(archive.read(matches[0]).decode("utf-8"))
    if manifest.get("version") != VERSION:
        raise ValueError(f"Unexpected engineering bundle version in {manifest_name}: {manifest.get('version')}")
    listed = manifest.get("files")
    if not isinstance(listed, list):
        raise ValueError(f"Embedded manifest file table is missing: {manifest_name}")
    expected: dict[str, dict] = {}
    for entry in listed:
        relative = entry.get("path")
        safe_member_parts(relative)
        if relative in expected:
            raise ValueError(f"Duplicate path in embedded manifest: {relative}")
        expected[relative] = entry
    if expected.keys() != actual_records.keys():
        missing = sorted(expected.keys() - actual_records.keys())
        unexpected = sorted(actual_records.keys() - expected.keys())
        raise ValueError(f"Embedded manifest member set differs (missing={missing[:4]}, unexpected={unexpected[:4]})")
    for relative, record in actual_records.items():
        declared = expected[relative]
        if declared.get("size") != record["size"] or declared.get("sha256") != record["sha256"]:
            raise ValueError(f"Embedded manifest hash/size mismatch for {relative}")


def process_bundle(path: Path, edition: str, expected_name: str, expected_root: str,
                   stage: Path, wheel_path: Path, wheel_sha: str, wheel_size: int,
                   old_index: dict[str, dict], source_prefix: str,
                   locked_hashes: set[str]) -> dict:
    path = ensure_project_file(path, f"built {edition} engineering ZIP")
    if path.name != expected_name:
        raise ValueError(f"Expected {expected_name}, got {path.name}")
    archive_sha = sha256_file(path)
    archive_size = path.stat().st_size
    verify_zip_sidecar(path, archive_sha)
    blueprint_entries: list[dict] = []
    actual_records: dict[str, dict] = {}
    seen_destinations: set[str] = set()
    seen_names: set[str] = set()
    core_wheel_count = 0
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        if not infos:
            raise ValueError(f"Built ZIP is empty: {path}")
        for info in infos:
            parts = safe_member_parts(info.filename)
            if len(parts) < 2 or parts[0] != expected_root:
                raise ValueError(f"Unexpected member root in {path.name}: {info.filename}")
            if info.is_dir():
                raise ValueError(f"Unexpected directory entry in built engineering ZIP: {info.filename}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Refusing symbolic-link ZIP member: {info.filename}")
            if info.filename.casefold() in seen_names:
                raise ValueError(f"Duplicate case-insensitive ZIP member: {info.filename}")
            seen_names.add(info.filename.casefold())
            relative = PurePosixPath(*parts[1:]).as_posix()
            is_wheel = relative.lower().endswith(".whl")
            is_project_wheel = is_wheel and PurePosixPath(relative).name.lower() == WHEEL_NAME.lower()

            if is_project_wheel:
                core_wheel_count += 1
                digest, size = hash_member(archive, info)
                if digest != wheel_sha or size != wheel_size:
                    raise ValueError(f"Embedded 0.8.0 wheel differs from the top-level wheel: {info.filename}")
                source = f"{source_prefix}/vendor/{WHEEL_NAME}"
            elif is_wheel:
                digest, size = hash_member(archive, info)
                source = old_source_for(digest, size, old_index)
                if source is None and digest not in locked_hashes:
                    raise ValueError(
                        f"Source-less {edition} wheel is absent from the copied exact lockfile: {info.filename}"
                    )
            else:
                target_relative = PurePosixPath("files", edition, *parts[1:])
                target_key = target_relative.as_posix().casefold()
                if target_key in seen_destinations:
                    raise ValueError(f"Case-insensitive file-store collision for {info.filename}")
                seen_destinations.add(target_key)
                target = stage.joinpath(*target_relative.parts)
                digest, size = stream_member_to_file(archive, info, target)
                source = f"{source_prefix}/{target_relative.as_posix()}"
                if relative != "manifest.json":
                    actual_records[relative] = {"sha256": digest, "size": size}

            if is_wheel:
                actual_records[relative] = {"sha256": digest, "size": size}
            blueprint_entries.append(entry_blueprint(info, digest, size, source))

        validate_embedded_manifest(archive, infos, expected_root, actual_records)
    if core_wheel_count != 1:
        raise ValueError(f"Expected exactly one {WHEEL_NAME} member in {path.name}, found {core_wheel_count}")
    return {
        "edition": edition,
        "archive": expected_name,
        "sha256": archive_sha,
        "size": archive_size,
        "entries": blueprint_entries,
    }


def bundle_member_map(archive_path: Path) -> tuple[zipfile.ZipFile, dict[str, zipfile.ZipInfo]]:
    archive = zipfile.ZipFile(archive_path, "r")
    names = {item.filename: item for item in archive.infolist() if not item.is_dir()}
    return archive, names


def build_templates_and_guides_zip(stage: Path, windows_zip: Path, linux_zip: Path) -> tuple[Path, bytes]:
    output = stage / "assets" / "engineering-templates-and-guides.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    windows, win_names = bundle_member_map(windows_zip)
    linux, linux_names = bundle_member_map(linux_zip)
    try:
        win_root = PurePosixPath(next(iter(win_names))).parts[0]
        linux_root = PurePosixPath(next(iter(linux_names))).parts[0]
        payload: dict[str, bytes] = {}
        selected = {f"{win_root}/{name}" for name in APPLICATION_GUIDES}
        selected.update(name for name in win_names if name.startswith(f"{win_root}/examples/applications/"))
        selected.update(f"{win_root}/{name}" for name in APPLICATION_REQUIREMENTS if f"{win_root}/{name}" in win_names)
        selected.update(name for name in win_names if name == f"{win_root}/LICENSE" or name == f"{win_root}/NOTICE")
        selected.update(name for name in win_names if name.startswith(f"{win_root}/licenses/"))
        if not any(name.startswith(f"{win_root}/examples/applications/") for name in selected):
            raise ValueError("The engineering bundle has no examples/applications templates")
        for relative in APPLICATION_GUIDES:
            archive_member = f"{win_root}/{relative}"
            if archive_member not in win_names:
                raise ValueError(f"Missing required guide in built bundle: {relative}")
            selected.add(archive_member)
        for member in sorted(selected):
            relative = member[len(win_root) + 1:]
            linux_member = f"{linux_root}/{relative}"
            if member not in win_names or linux_member not in linux_names:
                raise ValueError(f"Selected guide/template is missing from one platform bundle: {relative}")
            win_data = windows.read(win_names[member])
            if win_data != linux.read(linux_names[linux_member]):
                raise ValueError(f"Guide/template differs between Windows and Linux bundles: {relative}")
            payload[relative] = win_data
        app_guide = payload["docs/APPLICATIONS_GUIDE.md"]
        (stage / "assets" / "APPLICATIONS_GUIDE.zh-CN.md").write_bytes(app_guide)
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for relative, data in sorted(payload.items(), key=lambda item: item[0].casefold()):
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
        return output, app_guide
    finally:
        windows.close()
        linux.close()


def artifact_record(path: Path) -> dict:
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_engineering_report(path: Path, records: list[dict], template_record: dict) -> None:
    rows = "\n".join(
        f"| `{item['name']}` | {item['bytes']} | `{item['sha256']}` |"
        for item in records
    )
    content = (
        "# unified-marl 0.8.0 工程分发报告\n\n"
        "本报告对应已准备的 Windows/Linux 工程安装包、通用 wheel 与 sdist。"
        "安装包内的 `manifest.json` 已逐文件核对 SHA-256 和字节数；"
        "发布准备器只处理既有构建产物，不下载依赖、不安装环境、不运行应用或训练。\n\n"
        "## 核心发布物\n\n"
        "| 文件 | 字节数 | SHA-256 |\n|---|---:|---|\n"
        f"{rows}\n\n"
        "## 工程模板与功能依赖\n\n"
        "`engineering-templates-and-guides.zip` 汇集应用模板、指南及随包功能需求文件；"
        f"其 SHA-256 为 `{template_record['sha256']}`，大小为 {template_record['bytes']} 字节。\n\n"
        "基础安装只使用包内核心依赖。仓储 MFVFD 需要 HMF；RODE-HMF 还需要 role 组件。"
        "交通仿真依赖单独构建/安装 CityFlow；电网仿真依赖 pandapower 等功能包并需联网安装。"
        "奖励模型的 Linux `hf-xet` 补充项只在用户选择 reward 功能时由目标环境联网安装。\n\n"
        "安装与运行命令见包内 `docs/ENGINEERING_DISTRIBUTION.md` 和各应用指南。"
        "所有安装脚本均不自动启动训练。\n"
    )
    path.write_text(content, encoding="utf-8", newline="\n")


def make_supplemental_assets(stage: Path, windows_zip: Path, linux_zip: Path,
                             primary_records: list[dict]) -> list[dict]:
    template_zip, _ = build_templates_and_guides_zip(stage, windows_zip, linux_zip)
    template_record = artifact_record(template_zip)
    asset_dir = stage / "assets"
    report_path = asset_dir / "ENGINEERING_REPORT.md"
    write_engineering_report(report_path, primary_records, template_record)

    report_record = artifact_record(report_path)
    manifest_assets = [*primary_records, artifact_record(asset_dir / "APPLICATIONS_GUIDE.zh-CN.md"),
                       template_record, report_record]
    delivery_manifest = {
        "release": "unified-marl 0.8.0 engineering",
        "format": 1,
        "artifacts": manifest_assets,
        "verification": {
            "bundle_member_manifests_checked": True,
            "installer_runs_performed": False,
            "training_or_experiments_run": False,
            "dependency_downloads_performed": False,
        },
        "dependency_scope": {
            "base": "core dependencies are bundled in each platform ZIP",
            "warehouse": "HMF is installed only when the warehouse feature is selected; RODE-HMF also needs role",
            "traffic": "CityFlow is an external source/build dependency",
            "powergrid": "pandapower and related application dependencies are installed by explicit feature selection",
            "reward_linux": "hf-xet is a recipient-side optional network supplement, not a bundled wheel",
        },
    }
    delivery_manifest_path = asset_dir / "DELIVERY_MANIFEST.json"
    json_write(delivery_manifest_path, delivery_manifest)

    checksum_paths = [
        stage / WINDOWS_ARCHIVE,
        stage / LINUX_ARCHIVE,
        stage / "vendor" / WHEEL_NAME,
        stage / "vendor" / SDIST_NAME,
        asset_dir / "APPLICATIONS_GUIDE.zh-CN.md",
        template_zip,
        report_path,
        delivery_manifest_path,
    ]
    sums = "".join(f"{sha256_file(item)}  {item.name}\n" for item in sorted(checksum_paths, key=lambda p: p.name.casefold()))
    (asset_dir / "SHA256SUMS.txt").write_text(sums, encoding="ascii", newline="\n")

    release_paths = [*checksum_paths, asset_dir / "SHA256SUMS.txt"]
    return [artifact_record(path) for path in release_paths]


def prepare(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    delivery_dir = (ROOT / "delivery").resolve()
    if delivery_dir == ROOT.resolve() or not delivery_dir.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Refusing release preparation outside the project workspace: {delivery_dir}")
    if output == delivery_dir or not output.is_relative_to(delivery_dir):
        raise ValueError(f"Release preparation output must be a child of delivery/: {output}")
    if output.exists():
        raise FileExistsError(f"Engineering release directory already exists; refusing to overwrite: {output}")

    windows_path = ensure_project_file(args.windows_zip, "Windows engineering ZIP")
    linux_path = ensure_project_file(args.linux_zip, "Linux engineering ZIP")
    wheel_path = ensure_project_file(args.wheel, "0.8.0 project wheel")
    sdist_path = ensure_project_file(args.sdist, "0.8.0 source distribution")
    old_blueprint_path = ensure_project_file(args.old_blueprint, "previous release blueprint")
    old_requirements = {
        edition: ensure_project_file(ROOT / relative, f"previous {edition} exact requirements")
        for edition, relative in REQUIREMENTS_FILES.items()
    }
    build_report_path = ensure_project_file(args.build_report, "engineering build report")

    wheel_sha = sha256_file(wheel_path)
    wheel_size = wheel_path.stat().st_size
    if wheel_path.name != WHEEL_NAME:
        raise ValueError(f"Expected project wheel {WHEEL_NAME}, got {wheel_path.name}")
    if sdist_path.name != SDIST_NAME:
        raise ValueError(f"Expected source distribution {SDIST_NAME}, got {sdist_path.name}")

    build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
    if build_report.get("status") != "BUILT" or build_report.get("version") != VERSION:
        raise ValueError("The engineering build report is not a completed 0.8.0 build")
    if build_report.get("wheel_sha256") != wheel_sha:
        raise ValueError("The engineering build report wheel hash differs from the supplied wheel")

    old_index = old_wheel_index(old_blueprint_path)
    locked_hashes = {
        edition: {
            match.group(1).lower()
            for match in REQUIREMENT_HASH.finditer(path.read_text(encoding="utf-8"))
        }
        for edition, path in old_requirements.items()
    }
    source_prefix = output.relative_to(ROOT).as_posix()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".engineering-release-prepare-", dir=delivery_dir))
    try:
        (temporary / ".gitattributes").write_text("# Preserve exact release input bytes on checkout.\n* -text\n",encoding="utf-8",newline="\n")
        (temporary / "assets").mkdir()
        (temporary / "files").mkdir()
        (temporary / "vendor").mkdir()
        shutil.copyfile(wheel_path, temporary / "vendor" / WHEEL_NAME)
        shutil.copyfile(sdist_path, temporary / "vendor" / SDIST_NAME)
        for edition, source in old_requirements.items():
            shutil.copyfile(source, temporary / f"requirements-{edition}.txt")

        wheel_record = artifact_record(temporary / "vendor" / WHEEL_NAME)
        sdist_record = artifact_record(temporary / "vendor" / SDIST_NAME)
        if wheel_record["sha256"] != wheel_sha or wheel_record["bytes"] != wheel_size:
            raise ValueError("Copied 0.8.0 wheel failed the SHA-256/size check")

        windows_bundle = process_bundle(
            windows_path, "windows", WINDOWS_ARCHIVE, Path(WINDOWS_ARCHIVE).stem,
            temporary, wheel_path, wheel_sha, wheel_size, old_index, source_prefix, locked_hashes["windows"],
        )
        linux_bundle = process_bundle(
            linux_path, "linux", LINUX_ARCHIVE, Path(LINUX_ARCHIVE).stem,
            temporary, wheel_path, wheel_sha, wheel_size, old_index, source_prefix, locked_hashes["linux"],
        )
        build_bundles = {item.get("target"): item for item in build_report.get("bundles", [])}
        for edition, bundle, expected in (
            ("windows", windows_bundle, WINDOWS_ARCHIVE),
            ("linux", linux_bundle, LINUX_ARCHIVE),
        ):
            report_item = build_bundles.get(edition)
            if not report_item or report_item.get("sha256") != bundle["sha256"] or report_item.get("size") != bundle["size"]:
                raise ValueError(f"Engineering build report differs from the {edition} ZIP")
            shutil.copyfile(windows_path if edition == "windows" else linux_path, temporary / expected)

        primary_records = [
            artifact_record(temporary / WINDOWS_ARCHIVE),
            artifact_record(temporary / LINUX_ARCHIVE),
            wheel_record,
            sdist_record,
        ]
        release_assets = make_supplemental_assets(temporary, windows_path, linux_path, primary_records)

        blueprint = {
            "bundles": [windows_bundle, linux_bundle],
        }
        json_write(temporary / "blueprint.json", blueprint)
        json_write(temporary / "asset-manifest.json", release_assets)

        final_blueprint = temporary / "blueprint.json"
        final_asset_manifest = temporary / "asset-manifest.json"
        result = {
            "status": "PREPARED",
            "version": VERSION,
            "output": str(output),
            "blueprint": str(output / final_blueprint.name),
            "asset_manifest": str(output / final_asset_manifest.name),
            "assets": release_assets,
            "wheel_source_mappings": sum(
                1 for bundle in blueprint["bundles"] for item in bundle["entries"]
                if item["filename"].lower().endswith(".whl") and "source" in item
            ),
            "archive_entries": {item["edition"]: len(item["entries"]) for item in blueprint["bundles"]},
            "downloads_performed": False,
            "installers_or_training_run": False,
        }
        os.replace(temporary, output)
        return result
    finally:
        if temporary.exists():
            resolved_delivery = delivery_dir.resolve()
            resolved_temporary = temporary.resolve()
            if resolved_temporary == resolved_delivery or not resolved_temporary.is_relative_to(resolved_delivery):
                raise RuntimeError(f"Refusing temporary cleanup outside delivery/: {resolved_temporary}")
            shutil.rmtree(temporary, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-zip", type=Path,
                        default=ROOT / "dist/internal" / WINDOWS_ARCHIVE)
    parser.add_argument("--linux-zip", type=Path,
                        default=ROOT / "dist/internal" / LINUX_ARCHIVE)
    parser.add_argument("--wheel", type=Path, default=ROOT / "dist" / WHEEL_NAME)
    parser.add_argument("--sdist", type=Path, default=ROOT / "dist" / SDIST_NAME)
    parser.add_argument("--build-report", type=Path,
                        default=ROOT / "dist/internal/engineering-delivery-build.json")
    parser.add_argument("--old-blueprint", type=Path, default=ROOT / "delivery/release/blueprint.json")
    parser.add_argument("--output", type=Path, default=ROOT / "delivery/engineering-release")
    args = parser.parse_args()
    try:
        result = prepare(args)
    except (OSError, RuntimeError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"Engineering release preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

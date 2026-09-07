"""Build and package standalone release artifacts for SIGANALYZER using PyInstaller."""

import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile


def get_version(workspace_root: Path) -> str:
    """Extract version from environment variable or siganalyzer package."""
    env_ver = os.environ.get("RELEASE_VERSION")
    if env_ver:
        return env_ver.lstrip("v").strip()
    init_py = workspace_root / "src" / "siganalyzer" / "__init__.py"
    if init_py.exists():
        with open(init_py, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"


def get_arch() -> str:
    """Normalize CPU architecture name to standard format."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return "x64"


def calculate_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def package_macos_dmg(workspace_root: Path, version: str) -> list[Path]:
    """Package SIGANALYZER.app into a compressed .dmg."""
    dist_dir = workspace_root / "dist"
    app_path = dist_dir / "SIGANALYZER.app"

    if not app_path.exists():
        print(f"Error: {app_path} does not exist. Cannot build DMG.", file=sys.stderr)
        return []

    arch = get_arch()
    dmg_name = f"SIGANALYZER-v{version}-macos-{arch}.dmg"
    dmg_path = dist_dir / dmg_name

    if dmg_path.exists():
        dmg_path.unlink()

    print(f"\n[macOS] Packaging {app_path.name} -> {dmg_name} via hdiutil...")
    cmd = [
        "hdiutil",
        "create",
        "-volname", "SIGANALYZER",
        "-srcfolder", str(app_path),
        "-ov",
        "-format", "UDZO",
        str(dmg_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"DMG successfully created: {dmg_path} ({dmg_path.stat().st_size:,} bytes)")
        return [dmg_path]
    except subprocess.CalledProcessError as e:
        print(f"Error creating DMG: {e.stderr.decode()}", file=sys.stderr)
        return []


def find_iscc() -> str | None:
    """Locate Inno Setup compiler executable on Windows."""
    which_iscc = shutil.which("iscc") or shutil.which("ISCC")
    if which_iscc:
        return which_iscc

    common_paths = [
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Inno Setup 6" / "ISCC.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Inno Setup 6" / "ISCC.exe",
        Path(os.environ.get("LocalAppData", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe",
    ]
    for p in common_paths:
        if p.exists():
            return str(p)
    return None


def package_windows(workspace_root: Path, version: str) -> list[Path]:
    """Package Windows build into installer executable or portable archive."""
    dist_dir = workspace_root / "dist"
    app_dir = dist_dir / "siganalyzer"

    if not app_dir.exists():
        print(f"Error: {app_dir} does not exist. Cannot package Windows build.", file=sys.stderr)
        return []

    arch = "x64"
    artifacts: list[Path] = []
    iscc_bin = find_iscc()

    if iscc_bin:
        print(f"\n[Windows] Found Inno Setup compiler: {iscc_bin}")
        installer_name = f"SIGANALYZER-v{version}-windows-{arch}.exe"
        installer_output = dist_dir / installer_name
        if installer_output.exists():
            installer_output.unlink()

        iss_content = f"""[Setup]
AppName=SIGANALYZER
AppVersion={version}
AppPublisher=SIGANALYZER Team
DefaultDirName={{autopf}}\\SIGANALYZER
DefaultGroupName=SIGANALYZER
OutputDir={dist_dir.resolve()}
OutputBaseFilename=SIGANALYZER-v{version}-windows-{arch}
Compression=lzma2/max
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
PrivilegesRequired=lowest

[Files]
Source: "{app_dir.resolve()}\\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{{autoprograms}}\\SIGANALYZER"; Filename: "{{app}}\\siganalyzer.exe"
Name: "{{autodesktop}}\\SIGANALYZER"; Filename: "{{app}}\\siganalyzer.exe"; Tasks: desktopicon

[Tasks]
Name: desktopicon; Description: "{{cm:CreateDesktopIcon}}"; GroupDescription: "{{cm:AdditionalIcons}}"

[Run]
Filename: "{{app}}\\siganalyzer.exe"; Description: "{{cm:LaunchProgram,SIGANALYZER}}"; Flags: nowait postinstall skipifsilent
"""
        iss_file = dist_dir / "installer.iss"
        with open(iss_file, "w", encoding="utf-8") as f:
            f.write(iss_content)

        print(f"Compiling Windows installer via Inno Setup: {installer_name}...")
        try:
            subprocess.run([iscc_bin, str(iss_file)], check=True)
            if installer_output.exists():
                print(f"Windows installer created: {installer_output} ({installer_output.stat().st_size:,} bytes)")
                artifacts.append(installer_output)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Inno Setup compilation failed ({e}). Falling back to ZIP archive.", file=sys.stderr)

    if not artifacts:
        zip_name = f"SIGANALYZER-v{version}-windows-{arch}.zip"
        zip_path = dist_dir / zip_name
        if zip_path.exists():
            zip_path.unlink()

        print(f"\n[Windows] Creating portable archive: {zip_name}...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(app_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = Path("SIGANALYZER") / file_path.relative_to(app_dir)
                    zf.write(file_path, arcname)
        print(f"Portable ZIP archive created: {zip_path} ({zip_path.stat().st_size:,} bytes)")
        artifacts.append(zip_path)

    return artifacts


def package_linux(workspace_root: Path, version: str) -> list[Path]:
    """Package Linux build into standalone tarball and AppImage if available."""
    dist_dir = workspace_root / "dist"
    app_dir = dist_dir / "siganalyzer"

    if not app_dir.exists():
        print(f"Error: {app_dir} does not exist. Cannot package Linux build.", file=sys.stderr)
        return []

    arch = get_arch()
    artifacts: list[Path] = []

    # 1. Always create standard portable .tar.gz
    tar_name = f"SIGANALYZER-v{version}-linux-{arch}.tar.gz"
    tar_path = dist_dir / tar_name
    if tar_path.exists():
        tar_path.unlink()

    print(f"\n[Linux] Creating portable tarball: {tar_name}...")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(app_dir, arcname=f"SIGANALYZER-v{version}-linux-{arch}")
    print(f"Linux tarball created: {tar_path} ({tar_path.stat().st_size:,} bytes)")
    artifacts.append(tar_path)

    # 2. If appimagetool is available, package AppImage
    appimagetool = shutil.which("appimagetool")
    if appimagetool:
        appimage_name = f"SIGANALYZER-v{version}-linux-{arch}.AppImage"
        appimage_path = dist_dir / appimage_name
        appdir = dist_dir / "AppDir"
        if appdir.exists():
            shutil.rmtree(appdir)
        appdir.mkdir(parents=True, exist_ok=True)

        print(f"[Linux] Found appimagetool. Preparing AppDir for {appimage_name}...")
        shutil.copytree(app_dir, appdir / "usr" / "bin", dirs_exist_ok=True)

        apprun = appdir / "AppRun"
        with open(apprun, "w", encoding="utf-8") as f:
            f.write("#!/bin/sh\nHERE=\"$(dirname \"$(readlink -f \"${0}\")\")\"\nexec \"$HERE/usr/bin/siganalyzer\" \"$@\"\n")
        apprun.chmod(0o755)

        desktop_file = appdir / "siganalyzer.desktop"
        with open(desktop_file, "w", encoding="utf-8") as f:
            f.write("[Desktop Entry]\nType=Application\nName=SIGANALYZER\nExec=siganalyzer\nIcon=siganalyzer\nCategories=Development;Engineering;\nTerminal=false\n")

        icon_src = workspace_root / "assets" / "icon.svg"
        if icon_src.exists():
            shutil.copy(icon_src, appdir / "siganalyzer.svg")
            shutil.copy(icon_src, appdir / ".DirIcon")

        try:
            env = os.environ.copy()
            env["ARCH"] = "x86_64" if arch == "x64" else arch
            subprocess.run([appimagetool, str(appdir), str(appimage_path)], check=True, env=env)
            if appimage_path.exists():
                print(f"AppImage successfully created: {appimage_path} ({appimage_path.stat().st_size:,} bytes)")
                artifacts.append(appimage_path)
        except Exception as e:
            print(f"Warning: AppImage generation failed: {e}", file=sys.stderr)

    return artifacts


def main() -> int:
    workspace_root = Path(__file__).resolve().parent.parent
    spec_path = workspace_root / "siganalyzer.spec"
    version = get_version(workspace_root)

    if not spec_path.exists():
        print(f"Error: Spec file not found at {spec_path}", file=sys.stderr)
        return 1

    try:
        import PyInstaller
    except ImportError:
        print("Error: PyInstaller is not installed in the current environment.", file=sys.stderr)
        return 1

    print(f"=== Building SIGANALYZER Standalone Release v{version} ===")
    print(f"Workspace: {workspace_root}")
    print(f"Platform:  {platform.system()} ({platform.machine()})")
    print(f"Spec file: {spec_path}")

    # Run PyInstaller
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        str(spec_path),
    ]

    try:
        subprocess.run(cmd, cwd=str(workspace_root), check=True)
    except subprocess.CalledProcessError as e:
        print(f"PyInstaller build failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode

    dist_dir = workspace_root / "dist"
    packaged_files: list[Path] = []

    # Platform specific packaging
    if sys.platform == "darwin":
        packaged_files.extend(package_macos_dmg(workspace_root, version))
    elif sys.platform == "win32":
        packaged_files.extend(package_windows(workspace_root, version))
    elif sys.platform.startswith("linux"):
        packaged_files.extend(package_linux(workspace_root, version))

    # Calculate SHA-256 for all generated distribution packages
    checksums: dict[str, str] = {}
    for p in packaged_files:
        if p.exists() and p.is_file():
            sha = calculate_sha256(p)
            checksums[p.name] = sha

    # Write SHA256SUMS.txt
    if checksums:
        checksums_file = dist_dir / "SHA256SUMS.txt"
        with open(checksums_file, "w", encoding="utf-8") as f:
            for fname, sha in sorted(checksums.items()):
                f.write(f"{sha}  {fname}\n")
                print(f"SHA-256 [{fname}]: {sha}")
        print(f"\nChecksums written to {checksums_file}")

    print("\n=== Release Build Completed Successfully ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

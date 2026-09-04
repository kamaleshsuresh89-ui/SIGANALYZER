"""Build standalone executable for SIGANALYZER using PyInstaller."""

from pathlib import Path
import subprocess
import sys


def main() -> int:
    workspace_root = Path(__file__).resolve().parent.parent
    spec_path = workspace_root / "siganalyzer.spec"

    if not spec_path.exists():
        print(f"Error: Spec file not found at {spec_path}", file=sys.stderr)
        return 1

    try:
        import PyInstaller
    except ImportError:
        print("Error: PyInstaller is not installed in the current environment.", file=sys.stderr)
        print("To build the standalone bundle, install pyinstaller ('pip install pyinstaller' or 'uv pip install pyinstaller')", file=sys.stderr)
        print("and re-run: python scripts/build_app.py", file=sys.stderr)
        return 1

    print(f"=== Building SIGANALYZER Standalone Bundle ===")
    print(f"Workspace: {workspace_root}")
    print(f"Spec file: {spec_path}")

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        str(spec_path),
    ]

    try:
        res = subprocess.run(cmd, cwd=str(workspace_root), check=True)
        print("\nBuild completed successfully.")
        dist_dir = workspace_root / "dist" / "siganalyzer"
        print(f"Artifacts located at: {dist_dir}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Build failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode
    except Exception as e:
        print(f"Build failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

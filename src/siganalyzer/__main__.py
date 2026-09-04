"""Main execution entry point for SIGANALYZER."""

import sys
from siganalyzer.app import run


def main() -> None:
    """Entry point when executed via `python -m siganalyzer` or console script."""
    sys.exit(run())


if __name__ == "__main__":
    main()

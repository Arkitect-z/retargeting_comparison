from __future__ import annotations

import argparse

from .interaction import run_interaction_native


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--case", choices=("box", "climb"), required=True)
    parser.add_argument("--variant", choices=("full", "no-hard"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    run_interaction_native(args.repo_root, args.case, args.variant, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

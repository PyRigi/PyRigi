import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = Path(__file__).resolve().parent

from _utils.cli import build_arg_parser, parse_and_resolve  # noqa: E402
from pipeline import run_benchmark_pipeline  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    config = parse_and_resolve(args, BENCHMARK_DIR)

    run_benchmark_pipeline(config)


if __name__ == "__main__":
    main()

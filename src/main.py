import argparse
from pathlib import Path

from builder import USQLModelBuilder
from query_engine import QueryEngine


def run_program(source: str) -> str:
    # Phase 1: build declarations/load state + parsed query objects.
    state = USQLModelBuilder().build(source)

    # Phase 2: execute queries in order and get the final relation result.
    result = QueryEngine().run(state)

    # Repr output already matches required USQL stdout format.
    return repr(result)


def run_optimizer(source: str) -> str:
    # TODO: implement query optimizer mode (--o flag).
    # Will print each optimisation step and the equivalence rule applied.
    raise NotImplementedError("Query optimizer not yet implemented.")


def main() -> None:
    # CLI entry point — dispatches to interpreter or optimizer based on flags.
    # Usage:
    #   python main.py program.usql        → interpret
    #   python main.py --o program.usql    → optimize (not yet implemented)
    parser = argparse.ArgumentParser(description="USQL interpreter")
    parser.add_argument("--o", action="store_true", help="Run the query optimiser instead of interpreting")
    parser.add_argument("program", help="Path to a USQL source file")
    args = parser.parse_args()

    source = Path(args.program).read_text(encoding="utf-8")

    if args.o:
        print(run_optimizer(source))
    else:
        print(run_program(source))


if __name__ == "__main__":
    main()

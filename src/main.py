import argparse
from pathlib import Path

from builder import USQLModelBuilder
from optimizer import QueryOptimizer
from query_engine import QueryEngine

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
    state = USQLModelBuilder().build(source)

    if args.o:
        print(QueryOptimizer().run(state))
    else:
        print(QueryEngine().run(state))


if __name__ == "__main__":
    main()

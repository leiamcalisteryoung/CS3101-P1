import argparse
from pathlib import Path

from builder import USQLModelBuilder
from query_engine import QueryEngine

def main() -> None:
    # CLI entry point: takes one USQL source file path.
    parser = argparse.ArgumentParser(description="USQL interpreter")
    parser.add_argument("program", help="Path to a USQL source file")
    args = parser.parse_args()

    source_path = Path(args.program)
    source = source_path.read_text(encoding="utf-8")
    output = run_program(source)
    print(output)

def run_program(source: str) -> str:
    # Phase 1: build declarations/load state + parsed query objects.
    state = USQLModelBuilder().build(source)

    # Phase 2: execute queries in order and get the final relation result.
    result = QueryEngine().run(state)

    # Repr output already matches required USQL stdout format.
    return repr(result)

if __name__ == "__main__":
    main()

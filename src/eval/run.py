"""Eval harness entry point.

Usage:
    python -m src.eval.run --scenarios src/eval/scenarios/
    python -m src.eval.run --scenarios src/eval/scenarios/ --output eval/results/my_run.md
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

from src.eval.report import write_report
from src.eval.runner import ScenarioResult, run_scenario
from src.eval.scenario import load_scenarios

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval harness against all scenarios.")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default="src/eval/scenarios/",
        help="Directory containing .yaml scenario files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown report path. Defaults to eval/results/run_TIMESTAMP.md.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    scenarios = load_scenarios(args.scenarios)
    if not scenarios:
        print(f"No scenarios found in {args.scenarios}")
        sys.exit(1)

    if args.output is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("eval/results") / f"run_{ts}.md"
    else:
        output_path = args.output

    results: list[ScenarioResult] = []
    try:
        for scenario in scenarios:
            print(f"Running: {scenario.name}")
            result = run_scenario(scenario)
            results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(f"  {status} ({result.duration_seconds:.1f}s)")
    except KeyboardInterrupt:
        print("\nInterrupted. Writing partial results...")

    if not results:
        print("No results to write.")
        sys.exit(0)

    write_report(results, output_path)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    print(f"\n{passed}/{total} passed. Report written to {output_path}")


if __name__ == "__main__":
    main()

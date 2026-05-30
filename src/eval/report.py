"""Markdown report writer for eval results."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from src.eval.runner import ScenarioResult


def write_report(results: list[ScenarioResult], output_path: Path) -> None:
    """Write a markdown report summarising the eval run to output_path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    pass_rate = passed / total if total else 0.0
    total_runtime = sum(r.duration_seconds for r in results)

    by_diagnosis: dict[str, list[ScenarioResult]] = defaultdict(list)
    for r in results:
        by_diagnosis[r.expected_diagnosis].append(r)

    passing_results = [r for r in results if r.passed]
    avg_confidence = (
        sum(r.confidence for r in passing_results if r.confidence is not None)
        / len(passing_results)
        if passing_results
        else 0.0
    )
    avg_tools = sum(len(r.tools_called) for r in results) / total if total else 0.0

    lines: list[str] = []

    lines.append("# Eval Run Summary\n")
    lines.append(f"- **Total scenarios:** {total}")
    lines.append(f"- **Passed:** {passed} / {total} ({pass_rate:.0%})")
    lines.append(f"- **Total runtime:** {total_runtime:.1f}s")
    lines.append(f"- **Avg tools per scenario:** {avg_tools:.1f}")
    lines.append(f"- **Avg confidence on passes:** {avg_confidence:.2f}")
    lines.append("")
    lines.append("**Per-diagnosis breakdown:**")
    for diag, group in sorted(by_diagnosis.items()):
        n_pass = sum(1 for r in group if r.passed)
        lines.append(f"- {diag}: {n_pass}/{len(group)}")
    lines.append("")

    lines.append("## Results Table\n")
    headers = [
        "name",
        "expected_diag",
        "actual_diag",
        "expected_action",
        "actual_action",
        "confidence",
        "tools",
        "result",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in results:
        confidence_str = f"{r.confidence:.2f}" if r.confidence is not None else "-"
        row = [
            r.scenario_name,
            r.expected_diagnosis,
            r.actual_diagnosis or "-",
            r.expected_action,
            r.actual_action or "-",
            confidence_str,
            str(len(r.tools_called)),
            "PASS" if r.passed else "FAIL",
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    failures = [r for r in results if not r.passed]
    if failures:
        lines.append("## Failure Details\n")
        for r in failures:
            lines.append(f"### {r.scenario_name}\n")
            lines.append(f"**Reason:** {r.failure_reason}\n")
            if r.reasoning_trace:
                lines.append("**Reasoning trace:**\n")
                lines.append("```")
                lines.append(r.reasoning_trace)
                lines.append("```\n")

    output_path.write_text("\n".join(lines) + "\n")

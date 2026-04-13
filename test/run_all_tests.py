import subprocess
import sys
from datetime import datetime
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent

TEST_SCRIPTS = [
    "test_data_indexing.py",
    "test_rag_unit_rules.py",
    "test_api_endpoints.py",
    "test_performance_metrics.py",
    "evaluate_rag.py",
]

REPORT_FILE = TEST_DIR / "test_report.md"


def run_test(script_name: str) -> tuple[str, int, str]:
    script_path = TEST_DIR / script_name
    cmd = [sys.executable, str(script_path)]

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return script_name, proc.returncode, output.strip()


def _build_report(results: list[tuple[str, int, str]]) -> str:
    lines: list[str] = []
    total = len(results)
    passed = [name for name, code, _ in results if code == 0]
    failed = [name for name, code, _ in results if code != 0]

    lines.append("# Test Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Summary")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| Total scripts | {total} |")
    lines.append(f"| Passed | {len(passed)} |")
    lines.append(f"| Failed | {len(failed)} |")
    lines.append("")

    if failed:
        lines.append("## Failed scripts")
        for name in failed:
            lines.append(f"- {name}")
        lines.append("")

    lines.append("## Details")
    lines.append("| Script | Result | Exit code |")
    lines.append("| --- | --- | ---: |")
    for name, code, _ in results:
        result = "PASSED" if code == 0 else "FAILED"
        lines.append(f"| {name} | {result} | {code} |")
    lines.append("")

    lines.append("## Outputs")
    for name, _, output in results:
        if output:
            lines.append(f"### {name}")
            lines.append("```text")
            lines.append(output)
            lines.append("```")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    print("Running full test suite...\n")

    results: list[tuple[str, int, str]] = []
    for script in TEST_SCRIPTS:
        print(f"=== {script} ===")
        name, code, output = run_test(script)
        if output:
            print(output)
        print(f"Result: {'PASSED' if code == 0 else 'FAILED'}\n")
        results.append((name, code, output))

    failed = [name for name, code, _ in results if code != 0]
    passed = [name for name, code, _ in results if code == 0]

    print("Summary")
    print(f"Passed: {len(passed)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")

    if failed:
        print("Failed tests:")
        for name in failed:
            print(f"- {name}")
        REPORT_FILE.write_text(_build_report(results), encoding="utf-8")
        print(f"\nTest report written to {REPORT_FILE}")
        return 1

    print("All tests passed")
    REPORT_FILE.write_text(_build_report(results), encoding="utf-8")
    print(f"\nTest report written to {REPORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

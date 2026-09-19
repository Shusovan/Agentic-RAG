import json
from datetime import datetime
from pathlib import Path

from evaluation.diagnostic_evaluator import DiagnosticEvaluator


BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"

K_VALUES = [1, 3, 5, 10]


def find_latest_retrieval_report() -> Path:
    """
    Find the most recent retrieval evaluation report.
    """

    reports = sorted(
        REPORT_DIR.glob("retrieval_evaluation_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not reports:
        raise FileNotFoundError(
            "No retrieval evaluation report found."
        )

    return reports[0]


def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # Find latest retrieval evaluation
    # ----------------------------------------------------------

    retrieval_report_path = find_latest_retrieval_report()

    print(
        f"Using retrieval report: "
        f"{retrieval_report_path.name}"
    )

    # ----------------------------------------------------------
    # Load retrieval evaluation
    # ----------------------------------------------------------

    with open(
        retrieval_report_path,
        "r",
        encoding="utf-8",
    ) as file:

        retrieval_evaluation = json.load(file)

    # ----------------------------------------------------------
    # Create diagnostic evaluator
    # ----------------------------------------------------------

    evaluator = DiagnosticEvaluator(
        k_values=K_VALUES,
    )

    # ----------------------------------------------------------
    # Run diagnostic evaluation
    # ----------------------------------------------------------

    diagnostic_report = evaluator.evaluate_dataset(
        retrieval_evaluation
    )

    # ----------------------------------------------------------
    # Save report
    # ----------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    output_path = (
        REPORT_DIR
        / f"diagnostic_evaluation_{timestamp}.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            diagnostic_report,
            file,
            indent=4,
        )

    # ----------------------------------------------------------
    # Print summary
    # ----------------------------------------------------------

    print("\nDiagnostic Evaluation Complete")
    print("--------------------------------")

    print(
        f"Total Queries: "
        f"{diagnostic_report['total_queries']}"
    )

    print(
        f"Successful Queries: "
        f"{diagnostic_report['successful_queries']}"
    )

    print(
        f"Failed Queries: "
        f"{diagnostic_report['failed_queries']}"
    )

    # ----------------------------------------------------------
    # Print issue counts
    # ----------------------------------------------------------

    print("\nIssue Counts:")

    issue_counts = diagnostic_report[
        "summary"
    ][
        "issue_counts"
    ]

    if issue_counts:

        for issue, count in issue_counts.items():
            print(
                f"  {issue}: {count}"
            )

    else:

        print("No issues detected.")


    # Report location
    print(f"\nReport saved to:\n{output_path}")


if __name__ == "__main__":
    main()
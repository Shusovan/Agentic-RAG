import json
import logging
from datetime import datetime
from pathlib import Path

from config.logs_file import setup_logging
from config.vector_dependency import (vector_store, embedding_pipeline)

from rag.retriever import Retriever
from evaluation.retrieval_evaluator import (RetrievalEvaluator)


setup_logging()

logger = logging.getLogger(__name__)


# path
BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (BASE_DIR / "datasets" / "retrieval_dataset.json")

REPORT_DIR = (BASE_DIR / "reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# config
K_VALUES = [1, 3, 5, 10]


# load
def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: "f"{DATASET_PATH}")

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


# save report to json
def save_json_report(report):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_path = (REPORT_DIR / f"retrieval_evaluation_{timestamp}.json")

    with output_path.open("w",encoding="utf-8") as file:
            json.dump( report, file, indent=4)

    return output_path


# main
def main():

    logger.info("Starting retrieval evaluation")

    dataset = load_dataset()

    logger.info("Loaded %s evaluation queries", len(dataset))

    # Create the SAME retriever used by the application
    retriever = Retriever(vector_store=vector_store, embedding_manager=embedding_pipeline)

    evaluator = RetrievalEvaluator(retriever=retriever)

    # Run evaluation
    report = evaluator.eval_dataset(dataset=dataset, k_values=K_VALUES)

    # Save report
    report_path = save_json_report(report)

    # Console summary
    print("\n")
    print("=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)
    print(f"Total queries:      "f"{report['total_queries']}")
    print(f"Successful queries: "f"{report['successful_queries']}")
    print(f"Failed queries:     "f"{report['failed_queries']}")
    print("-" * 60)

    for metric, value in (report["metrics"].items()):
        print(f"{metric:<20} "f"{value:.4f}")

    print("=" * 60)
    print(f"\nReport saved to:\n {report_path}")


if __name__ == "__main__":
    main()
import json
import os
import statistics
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVENTS_FILE = PROJECT_ROOT / "data" / "events.json"
FAISS_DIR = PROJECT_ROOT / "data" / "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Perf budgets (seconds), override with env vars if needed.
MAX_INDEX_LOAD_S = float(os.getenv("RAG_MAX_INDEX_LOAD_S", "8.0"))
MAX_QUERY_P95_S = float(os.getenv("RAG_MAX_QUERY_P95_S", "1.2"))

QUERY_SET = [
    "donne moi un evenement en 2026",
    "donne moi un evenement d'escrime en septembre 2025",
    "donne moi un evenement de musique en 2026",
    "donne moi un evenement cette semaine",
    "donne moi un evenement la semaine prochaine",
]


class ValidationError(Exception):
    pass


def _load_events_count() -> int:
    if not EVENTS_FILE.exists():
        raise ValidationError(f"Missing file: {EVENTS_FILE}")
    with EVENTS_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValidationError("events.json must contain a non-empty list")
    return len(data)


def _measure_retrieval_metrics() -> dict:
    start = time.perf_counter()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
    index_load_s = time.perf_counter() - start

    query_times = []
    hits = 0
    for q in QUERY_SET:
        t0 = time.perf_counter()
        docs = db.similarity_search(q, k=5)
        query_times.append(time.perf_counter() - t0)
        if docs:
            hits += 1

    p50 = statistics.median(query_times) if query_times else 0.0
    p95 = sorted(query_times)[max(0, int(len(query_times) * 0.95) - 1)] if query_times else 0.0

    return {
        "index_load_s": index_load_s,
        "query_p50_s": p50,
        "query_p95_s": p95,
        "queries": len(QUERY_SET),
        "queries_with_hits": hits,
    }


def main() -> int:
    try:
        events_count = _load_events_count()

        if not (FAISS_DIR / "index.faiss").exists():
            raise ValidationError(f"Missing FAISS index file: {FAISS_DIR / 'index.faiss'}")

        metrics = _measure_retrieval_metrics()

        print("Performance metrics")
        print(f"events_count: {events_count}")
        print(f"index_load_s: {metrics['index_load_s']:.4f}")
        print(f"query_p50_s: {metrics['query_p50_s']:.4f}")
        print(f"query_p95_s: {metrics['query_p95_s']:.4f}")
        print(f"queries_with_hits: {metrics['queries_with_hits']}/{metrics['queries']}")

        perf_passed = (
            metrics["index_load_s"] <= MAX_INDEX_LOAD_S
            and metrics["query_p95_s"] <= MAX_QUERY_P95_S
        )

        print(
            "Performance Gate: "
            f"{'PASSED' if perf_passed else 'FAILED'} "
            f"(max_index_load={MAX_INDEX_LOAD_S:.2f}s, max_query_p95={MAX_QUERY_P95_S:.2f}s)"
        )

        return 0 if perf_passed else 1
    except ValidationError as err:
        print(f"Performance metrics failed: {err}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

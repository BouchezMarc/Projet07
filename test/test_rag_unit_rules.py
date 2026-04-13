import importlib
import sys
import types
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class ValidationError(Exception):
    pass


class _FakeDB:
    def similarity_search(self, question: str, k: int = 5):
        return []


class _FakeFAISS:
    @staticmethod
    def load_local(*args, **kwargs):
        return _FakeDB()


class _FakeEmbeddings:
    def __init__(self, *args, **kwargs):
        pass


class _FakeMistral:
    def __init__(self, *args, **kwargs):
        self.chat = types.SimpleNamespace(complete=lambda **_: None)


def _inject_stubs():
    mod_vectorstores = types.ModuleType("langchain_community.vectorstores")
    mod_vectorstores.FAISS = _FakeFAISS

    mod_hf = types.ModuleType("langchain_huggingface")
    mod_hf.HuggingFaceEmbeddings = _FakeEmbeddings

    mod_mistral_client = types.ModuleType("mistralai.client")
    mod_mistral_client.Mistral = _FakeMistral

    mod_mistral = types.ModuleType("mistralai")
    mod_mistral.Mistral = _FakeMistral

    sys.modules["langchain_community.vectorstores"] = mod_vectorstores
    sys.modules["langchain_huggingface"] = mod_hf
    sys.modules["mistralai.client"] = mod_mistral_client
    sys.modules["mistralai"] = mod_mistral


def _assert(condition: bool, message: str):
    if not condition:
        raise ValidationError(message)


def _test_period_rules(rag):
    start, end = rag._question_period("donne moi un evenement en 2026")
    _assert(start == date(2026, 1, 1) and end == date(2026, 12, 31), "Year-only period rule failed")

    start, end = rag._question_period("donne moi un evenement en septembre 2025")
    _assert(start == date(2025, 9, 1) and end == date(2025, 9, 30), "Month+year period rule failed")

    start, end = rag._question_period("donne moi un evenement cette semaine")
    _assert(end >= start, "Current-week bounds invalid")

    start2, end2 = rag._question_period("donne moi un evenement la semaine prochaine")
    _assert(start2 > start and end2 > end, "Next-week bounds should be after current week")


def _test_topic_extraction(rag):
    terms = rag._extract_topic_terms("donne moi un evenement d'escrime en septembre 2025")
    _assert("escrime" in terms, "Topic extraction missed 'escrime'")
    _assert("septembre" not in terms, "Month should not be treated as topic term")


def _test_period_matching(rag):
    metadata = {"date_start": "2025-09-21", "date_end": "2025-09-21"}
    ok = rag._matches_period(metadata, date(2025, 9, 1), date(2025, 9, 30))
    _assert(ok, "Date overlap should match within September 2025")

    not_ok = rag._matches_period(metadata, date(2026, 9, 1), date(2026, 9, 30))
    _assert(not not_ok, "Date overlap should not match in September 2026")


def _test_dedup(rag):
    doc1 = types.SimpleNamespace(
        page_content="A",
        metadata={"title": "Escrime", "date_start": "2025-09-21", "date_end": "2025-09-21"},
    )
    doc2 = types.SimpleNamespace(
        page_content="B",
        metadata={"title": "Escrime", "date_start": "2025-09-21", "date_end": "2025-09-21"},
    )
    doc3 = types.SimpleNamespace(
        page_content="C",
        metadata={"title": "Autre", "date_start": "2025-09-22", "date_end": "2025-09-22"},
    )

    deduped = rag._deduplicate_docs([doc1, doc2, doc3])
    _assert(len(deduped) == 2, "Deduplication should remove one duplicate event")


def main() -> int:
    try:
        _inject_stubs()
        rag = importlib.import_module("app.rag")

        _test_period_rules(rag)
        _test_topic_extraction(rag)
        _test_period_matching(rag)
        _test_dedup(rag)

        print("RAG unit-rule tests passed")
        return 0
    except ValidationError as err:
        print(f"RAG unit-rule tests failed: {err}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

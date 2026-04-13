import json
import re
import sys
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVENTS_FILE = PROJECT_ROOT / "data" / "events.json"
FAISS_DIR = PROJECT_ROOT / "data" / "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

REQUIRED_DOC_KEYS = {"id", "text", "metadata"}
REQUIRED_METADATA_KEYS = {"title", "description", "city", "date_start", "date_end", "tags"}
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ValidationError(Exception):
    pass


def _load_events() -> list[dict]:
    if not EVENTS_FILE.exists():
        raise ValidationError(f"Missing file: {EVENTS_FILE}")

    with EVENTS_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValidationError("events.json must contain a non-empty list")

    return data


def _validate_event_schema(events: list[dict], sample_limit: int = 200) -> None:
    # Validate a bounded sample for quick checks while still catching shape errors.
    to_check = events[:sample_limit]

    for idx, event in enumerate(to_check):
        if not isinstance(event, dict):
            raise ValidationError(f"Event at index {idx} is not an object")

        missing_doc_keys = REQUIRED_DOC_KEYS - set(event.keys())
        if missing_doc_keys:
            raise ValidationError(f"Event at index {idx} is missing keys: {sorted(missing_doc_keys)}")

        if not str(event.get("id", "")).strip():
            raise ValidationError(f"Event at index {idx} has empty id")

        text = event.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValidationError(f"Event at index {idx} has empty text")

        metadata = event.get("metadata")
        if not isinstance(metadata, dict):
            raise ValidationError(f"Event at index {idx} has invalid metadata")

        missing_meta_keys = REQUIRED_METADATA_KEYS - set(metadata.keys())
        if missing_meta_keys:
            raise ValidationError(
                f"Event at index {idx} metadata missing keys: {sorted(missing_meta_keys)}"
            )

        for date_key in ("date_start", "date_end"):
            date_value = metadata.get(date_key)
            if date_value is None:
                continue
            if not isinstance(date_value, str) or not ISO_DATE_RE.match(date_value):
                raise ValidationError(
                    f"Event at index {idx} has invalid {date_key}: {date_value!r}"
                )


def _validate_faiss_index() -> int:
    index_file = FAISS_DIR / "index.faiss"
    if not index_file.exists():
        raise ValidationError(f"Missing FAISS index file: {index_file}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # Access internal docstore for a quick size check.
    doc_count = len(getattr(db.docstore, "_dict", {}))
    if doc_count <= 0:
        raise ValidationError("FAISS index loaded but contains no documents")

    return doc_count


def main() -> int:
    try:
        events = _load_events()
        _validate_event_schema(events)
        doc_count = _validate_faiss_index()

        print("Data indexing tests passed")
        print(f"events.json rows: {len(events)}")
        print(f"FAISS documents: {doc_count}")
        return 0
    except ValidationError as err:
        print(f"Data indexing tests failed: {err}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from app.prompt import build_prompt
import os
import re
import unicodedata
from pathlib import Path
from datetime import date, datetime, timedelta
from calendar import monthrange

try:
    from mistralai.client import Mistral
except ImportError:
    from mistralai import Mistral


def _load_project_env():
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_project_env()

# ==============================
# CONFIG
# ==============================

INDEX_PATH = "data/faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") or os.getenv("KEY_MISTRAL")

if not MISTRAL_API_KEY or MISTRAL_API_KEY == "YOUR_API_KEY":
    raise ValueError("Missing Mistral API key. Set MISTRAL_API_KEY in .env")

# ==============================
# INIT
# ==============================

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

db = FAISS.load_local(
    INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

client = Mistral(api_key=MISTRAL_API_KEY)

# ==============================
# RAG PIPELINE
# ==============================

MONTHS_FR = {
    "janvier": 1,
    "fevrier": 2,
    "février": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "aout": 8,
    "août": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "decembre": 12,
    "décembre": 12,
}

IGNORED_QUERY_TERMS = {
    "donne", "moi", "un", "une", "des", "de", "d", "du", "la", "le", "les",
    "en", "pour", "sur", "avec", "et", "ou", "event", "evenement", "evenements",
    "événement", "événements", "ce", "cette", "mois", "mois-ci", "moisci",
    "aujourd", "hui", "demain", "prochain", "prochaine", "semaine", "weekend",
    "week", "end", "annee", "année", "an", "dans", "quel", "quelle", "quels",
    "quelles", "il", "y", "a", "t", "on", "veux", "veut", "trouve", "trouver",
    "janvier", "fevrier", "février", "mars", "avril", "mai", "juin", "juillet",
    "aout", "août", "septembre", "octobre", "novembre", "decembre", "décembre",
}


def _normalize_text(text: str) -> str:
    normalized = text.lower().strip()
    normalized = normalized.replace("’", "'")
    normalized = "".join(
        ch for ch in unicodedata.normalize("NFKD", normalized)
        if not unicodedata.combining(ch)
    )
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = " ".join(normalized.split())
    return normalized


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    end = date(year, month, monthrange(year, month)[1])
    return start, end


def _week_bounds(today: date) -> tuple[date, date]:
    start = today - timedelta(days=today.weekday())
    end = start + timedelta(days=6)
    return start, end


def _next_week_bounds(today: date) -> tuple[date, date]:
    current_week_start, _ = _week_bounds(today)
    next_week_start = current_week_start + timedelta(days=7)
    next_week_end = next_week_start + timedelta(days=6)
    return next_week_start, next_week_end


def _upcoming_weekend_bounds(today: date) -> tuple[date, date]:
    days_until_saturday = (5 - today.weekday()) % 7
    saturday = today + timedelta(days=days_until_saturday)
    sunday = saturday + timedelta(days=1)
    return saturday, sunday


def _week_after_next_bounds(today: date) -> tuple[date, date]:
    current_week_start, _ = _week_bounds(today)
    week_after_next_start = current_week_start + timedelta(days=14)
    week_after_next_end = week_after_next_start + timedelta(days=6)
    return week_after_next_start, week_after_next_end


def _question_period(question: str) -> tuple[date, date] | None:
    q = _normalize_text(question)
    today = date.today()

    if "aujourd hui" in q:
        return today, today

    if "demain" in q:
        d = today + timedelta(days=1)
        return d, d

    if "cette semaine" in q or "semaine en cours" in q:
        return _week_bounds(today)

    if "semaine prochaine" in q or "la semaine prochaine" in q or "prochaine semaine" in q:
        return _next_week_bounds(today)

    if (
        "semaine d apres" in q
        or "la semaine d apres" in q
        or "semaine dapres" in q
        or "la semaine dapres" in q
    ):
        return _week_after_next_bounds(today)

    if "week end prochain" in q or "ce week end prochain" in q:
        upcoming_saturday, _ = _upcoming_weekend_bounds(today)
        next_week_reference = upcoming_saturday + timedelta(days=7)
        return _upcoming_weekend_bounds(next_week_reference)

    if "ce week end" in q or "week end" in q:
        return _upcoming_weekend_bounds(today)

    if "mois prochain" in q:
        year = today.year + (1 if today.month == 12 else 0)
        month = 1 if today.month == 12 else today.month + 1
        return _month_bounds(year, month)

    if "ce mois" in q or "mois ci" in q or "ce mois ci" in q:
        return _month_bounds(today.year, today.month)

    # Ex: "en avril 2026" or "avril 2026"
    m = re.search(r"(?:en\s+)?([a-zéûîôàèùç]+)\s+(20\d{2})", q)
    if m:
        month_name = m.group(1)
        year = int(m.group(2))
        month = MONTHS_FR.get(month_name)
        if month:
            return _month_bounds(year, month)

    # Ex: "2026"
    y = re.search(r"\b(20\d{2})\b", q)
    if y:
        year = int(y.group(1))
        return date(year, 1, 1), date(year, 12, 31)

    return None


def _parse_event_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _matches_period(metadata: dict, period_start: date, period_end: date) -> bool:
    event_start = _parse_event_date(metadata.get("date_start"))
    event_end = _parse_event_date(metadata.get("date_end")) or event_start
    if not event_start or not event_end:
        return False
    return event_start <= period_end and event_end >= period_start


def _extract_topic_terms(question: str) -> list[str]:
    q = _normalize_text(question)
    tokens = re.findall(r"[a-z0-9]+", q)
    terms = []
    for token in tokens:
        if len(token) < 4:
            continue
        if token in IGNORED_QUERY_TERMS:
            continue
        if token.isdigit():
            continue
        terms.append(token)
    # Keep order while removing duplicates.
    seen = set()
    return [t for t in terms if not (t in seen or seen.add(t))]


def _matches_topic(doc, topic_terms: list[str]) -> bool:
    if not topic_terms:
        return True

    metadata = doc.metadata or {}
    haystack = " ".join(
        [
            doc.page_content or "",
            str(metadata.get("title", "")),
            str(metadata.get("description", "")),
            str(metadata.get("tags", "")),
        ]
    )
    haystack = _normalize_text(haystack)
    return all(term in haystack for term in topic_terms)


def _deduplicate_docs(docs: list) -> list:
    deduped = []
    seen_keys = set()
    for doc in docs:
        metadata = doc.metadata or {}
        key = (
            str(metadata.get("title", "")).strip().lower(),
            str(metadata.get("date_start", "")).strip(),
            str(metadata.get("date_end", "")).strip(),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(doc)
    return deduped


def _is_negative_answer(answer: str) -> bool:
    normalized = _normalize_text(answer)
    return "aucun evenement trouve" in normalized

def answer_question(question: str) -> str:
    # 1. Recherche des documents pertinents
    period = _question_period(question)
    topic_terms = _extract_topic_terms(question)
    docs = db.similarity_search(question, k=120 if period else 40)

    if period:
        period_start, period_end = period
        docs = [doc for doc in docs if _matches_period(doc.metadata, period_start, period_end)]

    # Apply topical filtering only when the question contains a clear topic term.
    if topic_terms:
        docs = [doc for doc in docs if _matches_topic(doc, topic_terms)]

    docs = _deduplicate_docs(docs)

    docs = docs[:5]

    # 2. Construction du contexte
    context = "\n\n".join([doc.page_content for doc in docs])

    # 3. Prompt
    prompt = build_prompt(context, question)

    # 4. Appel LLM
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    answer = response.choices[0].message.content

    # Keep the LLM in charge, but force a second pass when it contradicts non-empty context.
    if docs and _is_negative_answer(answer):
        repair_prompt = f"""
Tu as répondu qu'aucun événement ne correspondait.
Pourtant, le contexte contient des événements déjà filtrés par date/thème.

Contexte:
{context}

Question:
{question}

Instruction stricte:
- Si le contexte contient au moins un événement pertinent, ne réponds jamais "Aucun événement trouvé pour cette période.".
- Propose les événements pertinents du contexte au format:
  Nom, Date, Lieu, Description.
- N'invente rien.
"""

        second_response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "user", "content": repair_prompt}
            ],
            temperature=0.1
        )
        answer = second_response.choices[0].message.content

    # 5. Retour réponse
    return answer, context
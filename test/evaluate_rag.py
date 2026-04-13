import json
import sys
import re
import unicodedata
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag import answer_question
import os

# ==============================
# CONFIG
# ==============================
TEST_FILE = Path(__file__).parent / "test_questions.json"
OUTPUT_FILE = Path(__file__).parent / "rag_evaluation_results.json"

# Quality gates (adjustable via env vars)
MIN_CORRECT_RATE = float(os.getenv("RAG_MIN_CORRECT_RATE", "0.60"))
MAX_INCORRECT_RATE = float(os.getenv("RAG_MAX_INCORRECT_RATE", "0.25"))

# Load Mistral API key from .env
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

# ==============================
# CHARGEMENT DES TESTS
# ==============================
with open(TEST_FILE, "r", encoding="utf-8") as f:
    tests = json.load(f)

results = []

# ==============================
# EVALUATION SIMPLE (3 NIVEAUX)
# ==============================

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )
    text = re.sub(r"[*_`#>\-:\[\]\(\)]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _grade_prediction(expected: str, predicted: str) -> str:
    expected_n = _normalize(expected)
    predicted_n = _normalize(predicted)

    if not expected_n and not predicted_n:
        return "correcte"
    if not expected_n:
        return "incorrecte"

    # For explicit negative expectations, keep strict behavior.
    if "aucun evenement trouve" in expected_n:
        return "correcte" if "aucun evenement trouve" in predicted_n else "incorrecte"

    if expected_n == predicted_n:
        return "correcte"

    # If a short expected key phrase is contained in a richer answer, count as correct.
    expected_token_count = len(expected_n.split())
    if expected_n in predicted_n and expected_token_count <= 4:
        return "correcte"

    if expected_n in predicted_n or predicted_n in expected_n:
        return "partiellement correcte"

    expected_tokens = set(expected_n.split())
    predicted_tokens = set(predicted_n.split())
    if not expected_tokens:
        return "incorrecte"

    overlap_ratio = len(expected_tokens & predicted_tokens) / len(expected_tokens)
    if overlap_ratio >= 0.6:
        return "partiellement correcte"
    return "incorrecte"


for test in tqdm(tests, desc="Evaluating RAG"):
    question = test["question"]
    expected = test.get("answer", "").strip()

    # Appel RAG
    rag_output = answer_question(question)
    predicted = rag_output[0].strip() if isinstance(rag_output, tuple) else str(rag_output).strip()

    # Evaluation 3 niveaux
    grade = _grade_prediction(expected, predicted)

    results.append({
        "question": question,
        "expected": expected,
        "predicted": predicted,
        "evaluation": grade
    })

# ==============================
# Résumé
# ==============================
total = len(results)
count_correcte = sum(1 for r in results if r["evaluation"] == "correcte")
count_partielle = sum(1 for r in results if r["evaluation"] == "partiellement correcte")
count_incorrecte = sum(1 for r in results if r["evaluation"] == "incorrecte")

if total > 0:
    correct_rate = count_correcte / total
    incorrect_rate = count_incorrecte / total

    print(f"Correcte: {correct_rate*100:.2f}% ({count_correcte}/{total})")
    print(f"Partiellement correcte: {count_partielle/total*100:.2f}% ({count_partielle}/{total})")
    print(f"Incorrecte: {incorrect_rate*100:.2f}% ({count_incorrecte}/{total})")

    quality_passed = (correct_rate >= MIN_CORRECT_RATE) and (incorrect_rate <= MAX_INCORRECT_RATE)
    print(
        "Quality Gate: "
        f"{'PASSED' if quality_passed else 'FAILED'} "
        f"(min_correct={MIN_CORRECT_RATE:.0%}, max_incorrect={MAX_INCORRECT_RATE:.0%})"
    )
else:
    print("Aucun test à évaluer.")
    quality_passed = False

# ==============================
# Sauvegarde
# ==============================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Evaluation results saved to {OUTPUT_FILE}")

if not quality_passed:
    raise SystemExit(1)
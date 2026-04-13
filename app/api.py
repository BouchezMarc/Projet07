import os
import sys
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from dotenv import load_dotenv

from app.rag import answer_question

# ==============================
# Charger variables .env
# ==============================
load_dotenv()
USERNAME = os.getenv("REBUILD_USER", "admin")
PASSWORD = os.getenv("REBUILD_PASSWORD", "password")

# ==============================
# FastAPI & sécurité
# ==============================
app = FastAPI(title="RAG OpenAgenda API")
security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != USERNAME or credentials.password != PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.username

# ==============================
# PROJECT_ROOT
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ==============================
# SCHEMA
# ==============================
class Query(BaseModel):
    question: str

# ==============================
# ROUTES PUBLIQUES
# ==============================
@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/ask")
def chat(query: Query):
    answer = answer_question(query.question)
    return {
        "question": query.question,
        "answer": answer
    }

# ==============================
# ENDPOINT PROTÉGÉ
# ==============================
@app.post("/rebuild")
def rebuild_knowledge_base(user: str = Depends(verify_credentials)):
    """
    Reconstruit la base de connaissances (fetch + build index)
    Protégé par Basic Auth
    """
    steps = [
        PROJECT_ROOT / "scripts" / "fetch_data.py",
        PROJECT_ROOT / "scripts" / "build_index.py",
    ]

    results = []

    for script in steps:
        child_env = os.environ.copy()
        child_env["PYTHONUTF8"] = "1"
        child_env["PYTHONIOENCODING"] = "utf-8"

        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
        )

        results.append(
            {
                "script": script.name,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        )

        if proc.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail={"message": "Rebuild failed", "steps": results}
            )

    return {"status": "ok", "message": "Knowledge base rebuilt", "steps": results}
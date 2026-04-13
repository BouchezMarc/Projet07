import json
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("✅ Running build_index.py")

# ==============================
# CONFIG
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "events.json"
INDEX_PATH = PROJECT_ROOT / "data" / "faiss_index"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ==============================
# LOAD DATA
# ==============================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Fichier d'entrée introuvable: {INPUT_FILE}. Lance d'abord scripts/fetch_data.py pour générer data/events.json."
    )

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    events = json.load(f)

texts = []
metadatas = []

for e in events:
    texts.append(e["text"])
    metadatas.append(e["metadata"])

# ==============================
# CHUNKING
# ==============================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunks = []
chunk_metadatas = []

for text, meta in zip(texts, metadatas):
    split_texts = splitter.split_text(text)
    chunks.extend(split_texts)
    chunk_metadatas.extend([meta] * len(split_texts))

print(f"📄 {len(chunks)} chunks générés")

# ==============================
# EMBEDDINGS + FAISS
# ==============================

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

db = FAISS.from_texts(
    texts=chunks,
    embedding=embeddings,
    metadatas=chunk_metadatas
)

# ==============================
# SAVE INDEX
# ==============================

os.makedirs(INDEX_PATH, exist_ok=True)
db.save_local(INDEX_PATH)

print(f"✅ Index sauvegardé dans {INDEX_PATH}")
import requests
import json
from datetime import datetime
from pathlib import Path
import re

# ==============================
# CONFIGURATION
# ==============================

DATASET = "evenements-publics-openagenda"
CITY_FILTER = "Valenciennes"
DEPARTMENT_FILTER = "Nord"

DATE_FROM = "2025-03-01"
DATE_TO = "2027-03-31"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "data" / "events.json"
ROWS = 100

# ==============================
# UTILITAIRES
# ==============================

def parse_date(date_str):
    """Parser la date sans l'heure, uniquement avec l'année, mois et jour"""
    if not date_str:
        return None
    try:
        # Ignorer l'heure et ne garder que la date (format YYYY-MM-DD)
        return datetime.strptime(date_str.split('T')[0], "%Y-%m-%d").date()
    except Exception:
        return None

def clean_text(text):
    """ Nettoyage du texte pour enlever les balises HTML """
    if not text:
        return ""
    text = re.sub("<.*?>", "", text)
    return text.strip()

# ==============================
# FETCH API
# ==============================

def fetch_events():
    url = "https://public.opendatasoft.com/api/records/1.0/search/"
    all_records = []
    start = 0

    while True:
        params = {
            "dataset": DATASET,
            "rows": ROWS,
            "start": start,
            "refine.location_city": CITY_FILTER,
            "refine.location_department": DEPARTMENT_FILTER
        }

        response = requests.get(url, params=params)
        data = response.json()
        records = data.get("records", [])

        if not records:
            break

        all_records.extend(records)
        start += ROWS

    return all_records

# ==============================
# FILTRE DATE
# ==============================

def filter_by_date(records):
    """Filtrer les événements uniquement en fonction de la date (année, mois, jour)"""
    filtered = []
    date_from = parse_date(DATE_FROM)
    date_to = parse_date(DATE_TO)

    for r in records:
        f = r.get("fields", {})
        start = parse_date(f.get("firstdate_begin"))
        end = parse_date(f.get("lastdate_end"))

        # Si on n'a pas de date de début, on passe cet événement
        if not start:
            continue

        # Si l'événement commence avant la plage mais se termine après
        if end:
            if (start <= date_to and end >= date_from):
                filtered.append(r)
        else:
            # Si l'événement ne possède pas de date de fin
            if start <= date_to:
                filtered.append(r)

    return filtered

# ==============================
# NORMALISATION
# ==============================

def normalize(record):
    f = record.get("fields", {})

    # Appliquer la fonction de formatage de date
    start_date = parse_date(f.get("firstdate_begin"))
    end_date = parse_date(f.get("lastdate_end"))

    # Récupérer et formater les tags
    tags = f.get("keywords_fr", [])
    
    # Si les tags sont une liste de caractères (type string), les réassembler en une chaîne
    if isinstance(tags, list) and all(isinstance(tag, str) for tag in tags):
        tags = ' '.join(tags)  # Joindre les éléments de la liste en une seule chaîne avec un espace
    
    return {
        "id": record.get("recordid"),
        "title": f.get("title_fr", ""),
        "description": clean_text(f.get("description_fr", "")),
        "date_start": start_date,  # Date sans heure
        "date_end": end_date,  # Date sans heure
        "city": f.get("location_city", ""),
        "department": f.get("location_department", ""),
        "tags": tags  # Tags sous forme de chaîne complète
    }
# ==============================
# TEXTE POUR EMBEDDING
# ==============================

def build_text(event):
    return f"""
Événement : {event['title']}
Description : {event['description']}
Lieu : {event['city']} ({event['department']})
Date : du {event['date_start']} au {event['date_end']}
Catégories : {', '.join(event['tags'])}
""".strip()

# ==============================
# MAIN
# ==============================

def main():
    print("📡 Récupération...")
    records = fetch_events()
    print(f"Total: {len(records)} événements récupérés.")

    print("📅 Filtrage...")
    records = filter_by_date(records)
    print(f"Après filtre: {len(records)}")

    print("🧱 Transformation...")
    events = [normalize(r) for r in records]

    vector_docs = []
    for e in events:
        vector_docs.append({
            "id": e["id"],
            "text": build_text(e),
            "metadata": {
                "title": e["title"],
                "description": e["description"],
                "city": e["city"],
                "date_start": e["date_start"].isoformat() if e["date_start"] else None,
                "date_end": e.get("date_end").isoformat() if e.get("date_end") else None,
                "tags": e["tags"]
            }
        })

    print("💾 Sauvegarde...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(vector_docs, f, ensure_ascii=False, indent=2)

    print("✅ Terminé")

if __name__ == "__main__":
    main()
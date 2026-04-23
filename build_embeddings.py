import json
import pandas as pd
from sentence_transformers import SentenceTransformer

CSV_PATH = "henkilolista.csv"
OUT_PATH = "embeddings.json"

def main():
    df = pd.read_csv(CSV_PATH)

    # Varmista sarakenimet: nimi, kuvaus
    if "nimi" not in df.columns or "kuvaus" not in df.columns:
        raise ValueError("CSV:stä puuttuu sarake 'nimi' tai 'kuvaus'")

    df["nimi"] = df["nimi"].astype(str).str.strip()
    df["kuvaus"] = df["kuvaus"].astype(str).str.strip()
    df = df[(df["nimi"] != "") & (df["kuvaus"] != "")].reset_index(drop=True)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = df["kuvaus"].tolist()
    names = df["nimi"].tolist()

    # normalize_embeddings=True -> cosine similarity = dot product (kun normit 1)
    embeddings = model.encode(texts, normalize_embeddings=True)

    payload = {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
        "items": [
            {"nimi": n, "emb": emb.tolist()}
            for n, emb in zip(names, embeddings)
        ],
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print(f"Wrote {OUT_PATH} with {payload['count']} items, dim={payload['dim']}")

if __name__ == "__main__":
    main()

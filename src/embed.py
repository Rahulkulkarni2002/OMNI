import json
import os
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

model = SentenceTransformer("all-MiniLM-L6-v2")


def setup_database(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS doc_chunks (
                id SERIAL PRIMARY KEY,
                chunk_id TEXT UNIQUE,
                title TEXT,
                url TEXT,
                text TEXT,
                embedding vector(384)
            );
        """)
    conn.commit()
    print("Database ready: doc_chunks table exists.")


def embed_and_store(chunks_path, conn):
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with conn.cursor() as cur:
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk["text"]).tolist()

            cur.execute("""
                INSERT INTO doc_chunks (chunk_id, title, url, text, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE
                SET title = EXCLUDED.title,
                    url = EXCLUDED.url,
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding;
            """, (
                chunk["chunk_id"],
                chunk["title"],
                chunk["url"],
                chunk["text"],
                embedding,
            ))

            if (i + 1) % 25 == 0:
                print(f"  embedded {i + 1}/{len(chunks)} chunks...")

    conn.commit()
    print(f"Done. Embedded and stored {len(chunks)} chunks.")


if __name__ == "__main__":
    conn = psycopg2.connect(DATABASE_URL)
    setup_database(conn)
    embed_and_store("../data/raw_docs/nim_chunks.json", conn)
    conn.close()

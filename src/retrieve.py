import os
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_search(query, conn, top_k=5):
    """Find chunks whose MEANING is closest to the query, using pgvector."""
    query_embedding = model.encode(query).tolist()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT chunk_id, title, url, text,
                   1 - (embedding <=> %s::vector) AS score
            FROM doc_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (query_embedding, query_embedding, top_k))
        rows = cur.fetchall()
    return [
        {"chunk_id": r[0], "title": r[1], "url": r[2], "text": r[3], "score": r[4]}
        for r in rows
    ]


def keyword_search(query, conn, top_k=5):
    """Find chunks containing the actual WORDS in the query, using Postgres full-text search."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT chunk_id, title, url, text,
                   ts_rank(to_tsvector('english', text), plainto_tsquery('english', %s)) AS score
            FROM doc_chunks
            WHERE to_tsvector('english', text) @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s;
        """, (query, query, top_k))
        rows = cur.fetchall()
    return [
        {"chunk_id": r[0], "title": r[1], "url": r[2], "text": r[3], "score": r[4]}
        for r in rows
    ]


def hybrid_search(query, conn, top_k=5):
    """Combine semantic + keyword results, remove duplicates, return the best matches."""
    semantic_results = semantic_search(query, conn, top_k=top_k)
    keyword_results = keyword_search(query, conn, top_k=top_k)

    combined = {}
    for r in semantic_results + keyword_results:
        if r["chunk_id"] not in combined:
            combined[r["chunk_id"]] = r

    results = sorted(combined.values(), key=lambda r: r["score"], reverse=True)
    return results[:top_k]

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


def query_model_catalog(model_name: str = None, required_gpu: str = None) -> str:
    """Query structured facts about NIM-supported models: required GPU, VRAM, precision, deployment type.
    Use this for specific model/GPU facts, not general explanations.

    Args:
        model_name: Optional partial model name to filter by (e.g. "Llama").
        required_gpu: Optional GPU name to filter by (e.g. "H100").
    """
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor() as cur:
        query = "SELECT model_name, required_gpu, min_vram_gb, supported_precision, deployment_type FROM model_catalog WHERE TRUE"
        params = []
        if model_name:
            query += " AND model_name ILIKE %s"
            params.append(f"%{model_name}%")
        if required_gpu:
            query += " AND required_gpu ILIKE %s"
            params.append(f"%{required_gpu}%")
        cur.execute(query, params)
        rows = cur.fetchall()
    conn.close()
    if not rows:
        return "No matching models found."
    return "\n".join(
        f"{r[0]}: requires {r[1]}, {r[2]}GB VRAM, {r[3]} precision, {r[4]} deployment" for r in rows
    )

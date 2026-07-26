import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

df = pd.read_csv("../data/model_catalog/model_catalog.csv")

conn = psycopg2.connect(DATABASE_URL)
with conn.cursor() as cur:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_catalog (
            model_name TEXT,
            required_gpu TEXT,
            min_vram_gb INTEGER,
            supported_precision TEXT,
            deployment_type TEXT
        );
    """)
    cur.execute("TRUNCATE model_catalog;")
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO model_catalog (model_name, required_gpu, min_vram_gb, supported_precision, deployment_type)
            VALUES (%s, %s, %s, %s, %s);
        """, (row['model_name'], row['required_gpu'], row['min_vram_gb'], row['supported_precision'], row['deployment_type']))
conn.commit()
conn.close()
print(f"Loaded {len(df)} models into model_catalog table.")

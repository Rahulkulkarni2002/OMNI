import os
import time
from dotenv import load_dotenv

load_dotenv()

print("1. Testing Postgres connection...")
t0 = time.time()
try:
    import psycopg2
    conn = psycopg2.connect(os.getenv("DATABASE_URL"), connect_timeout=10)
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM doc_chunks;")
    print(f"   OK - {cur.fetchone()[0]} chunks. ({time.time()-t0:.1f}s)")
    conn.close()
except Exception as e:
    print(f"   FAILED: {e} ({time.time()-t0:.1f}s)")

print("2. Testing embedding model...")
t0 = time.time()
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vec = model.encode("test query")
    print(f"   OK - embedding length {len(vec)}. ({time.time()-t0:.1f}s)")
except Exception as e:
    print(f"   FAILED: {e} ({time.time()-t0:.1f}s)")

print("3. Testing Gemini API directly (simple call, no tools)...")
t0 = time.time()
try:
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="Say hello in one word."
    )
    print(f"   OK - response: {response.text!r} ({time.time()-t0:.1f}s)")
except Exception as e:
    print(f"   FAILED: {e} ({time.time()-t0:.1f}s)")

print("Done.")

print("4. Testing Gemini WITH tools + automatic_function_calling disabled...")
t0 = time.time()
try:
    from google.genai import types

    def dummy_tool(query: str) -> str:
        """A test tool that just echoes back the query.

        Args:
            query: Any text.
        """
        return f"Echo: {query}"

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="Use the dummy_tool to echo the word 'test'.",
        config=types.GenerateContentConfig(
            tools=[dummy_tool],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    print(f"   OK - function_calls: {response.function_calls} ({time.time()-t0:.1f}s)")
except Exception as e:
    print(f"   FAILED: {e} ({time.time()-t0:.1f}s)")

print("Done.")

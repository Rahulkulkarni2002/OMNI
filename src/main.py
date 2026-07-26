import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
import psycopg2
from retrieve import hybrid_search

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
DATABASE_URL = os.getenv("DATABASE_URL")

app = FastAPI()


class QuestionRequest(BaseModel):
    question: str


class Source(BaseModel):
    title: str
    url: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/")
async def root():
    return {"status": "Omni backend is running"}


@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    conn = psycopg2.connect(DATABASE_URL)
    chunks = hybrid_search(request.question, conn, top_k=5)
    conn.close()

    if not chunks:
        return AnswerResponse(
            answer="I couldn't find anything relevant in the NIM documentation to answer that.",
            sources=[]
        )

    context = "\n\n---\n\n".join(
        f"[Source: {c['title']}]\n{c['text']}" for c in chunks
    )

    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say so honestly instead of guessing.

Context:
{context}

Question: {request.question}"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    seen = set()
    sources = []
    for c in chunks:
        if c["url"] not in seen:
            sources.append(Source(title=c["title"], url=c["url"]))
            seen.add(c["url"])

    return AnswerResponse(answer=response.text, sources=sources)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

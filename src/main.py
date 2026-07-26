import os
import psycopg2
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from retrieve import hybrid_search
from tools import query_model_catalog

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
    sources: list[Source] = []

    def search_documentation(query: str) -> str:
        """Search NVIDIA NIM/Triton documentation for explanations, concepts, or how-to steps.

        Args:
            query: The question or topic to search for in the documentation.
        """
        conn = psycopg2.connect(DATABASE_URL)
        chunks = hybrid_search(query, conn, top_k=5)
        conn.close()
        seen_urls = {s.url for s in sources}
        for c in chunks:
            if c["url"] not in seen_urls:
                sources.append(Source(title=c["title"], url=c["url"]))
                seen_urls.add(c["url"])
        if not chunks:
            return "No relevant documentation found."
        return "\n\n---\n\n".join(f"[{c['title']}]\n{c['text']}" for c in chunks)

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=request.question,
        config=types.GenerateContentConfig(
            tools=[search_documentation, query_model_catalog],
            system_instruction=(
                "You are Omni, an assistant for NVIDIA NIM and Triton. "
                "IMPORTANT: If the question mentions a specific model name, GPU, VRAM, memory, "
                "or precision requirement, you MUST call query_model_catalog first, even if you "
                "also plan to call search_documentation. Do not skip this tool for model-specific questions. "
                "Use search_documentation for conceptual or how-to questions. "
                "When in doubt, call BOTH tools rather than guessing which one is needed. "
                "Answer only using what the tools return -- never say information is missing "
                "without having actually called query_model_catalog first for any model-related question."
            ),
        ),
    )

    return AnswerResponse(answer=response.text, sources=sources)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

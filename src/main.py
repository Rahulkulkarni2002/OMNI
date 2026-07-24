import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


@app.get("/")
async def root():
    return {"status": "Omni backend is running"}


@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=f"Answer this question about NVIDIA NIM: {request.question}"
    )
    return AnswerResponse(answer=response.text)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

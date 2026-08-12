from fastapi import FastAPI
from pydantic import BaseModel
from agent import run_agent

app = FastAPI()


class QuestionRequest(BaseModel):
    question: str


class Source(BaseModel):
    title: str
    url: str


class ReasoningStep(BaseModel):
    tool: str
    detail: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source]
    reasoning: list[ReasoningStep]


@app.get("/")
async def root():
    return {"status": "Omni backend is running"}


@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    result = run_agent(request.question)
    return AnswerResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
        reasoning=[ReasoningStep(**r) for r in result["reasoning"]],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

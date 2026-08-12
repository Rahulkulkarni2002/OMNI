import os
import psycopg2
from dotenv import load_dotenv
from google import genai
from google.genai import types
from retrieve import hybrid_search
from tools import query_model_catalog as _query_model_catalog

load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=45000),
)
DATABASE_URL = os.getenv("DATABASE_URL")

SYSTEM_INSTRUCTION = (
    "You are Omni, an assistant for NVIDIA NIM and Triton. "
    "IMPORTANT: If the question mentions a specific model name, GPU, VRAM, memory, "
    "or precision requirement, you MUST call query_model_catalog first, even if you "
    "also plan to call search_documentation. Do not skip this tool for model-specific questions. "
    "Use search_documentation for conceptual or how-to questions. "
    "When in doubt, call BOTH tools rather than guessing which one is needed. "
    "Answer only using what the tools return."
)


def run_agent(question: str, max_turns: int = 5) -> dict:
    """Runs the full ReAct loop for one question. Returns answer, sources, reasoning,
    and the raw retrieved context text (needed for evaluation)."""
    sources = []
    reasoning = []
    retrieved_contexts = []  # raw text chunks, for evaluation

    def search_documentation(query: str) -> str:
        """Search NVIDIA NIM/Triton documentation for explanations, concepts, or how-to steps.

        Args:
            query: The question or topic to search for in the documentation.
        """
        conn = psycopg2.connect(DATABASE_URL)
        chunks = hybrid_search(query, conn, top_k=5)
        conn.close()
        seen_urls = {s["url"] for s in sources}
        for c in chunks:
            if c["url"] not in seen_urls:
                sources.append({"title": c["title"], "url": c["url"]})
                seen_urls.add(c["url"])
            retrieved_contexts.append(c["text"])
        if not chunks:
            return "No relevant documentation found."
        return "\n\n---\n\n".join(f"[{c['title']}]\n{c['text']}" for c in chunks)

    def query_model_catalog(model_name: str = None, required_gpu: str = None) -> str:
        """Query structured facts about NIM-supported models: required GPU, VRAM, precision, deployment type.
        Use this for specific model/GPU facts, not general explanations.

        Args:
            model_name: Optional partial model name to filter by (e.g. "Llama").
            required_gpu: Optional GPU name to filter by (e.g. "H100").
        """
        result = _query_model_catalog(model_name=model_name, required_gpu=required_gpu)
        retrieved_contexts.append(result)
        return result

    tool_functions = {
        "search_documentation": search_documentation,
        "query_model_catalog": query_model_catalog,
    }

    config = types.GenerateContentConfig(
        tools=[search_documentation, query_model_catalog],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        system_instruction=SYSTEM_INSTRUCTION,
    )

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    final_answer = "I couldn't determine an answer within the allowed reasoning steps."

    for turn in range(max_turns):
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=contents,
            config=config,
        )

        if not response.function_calls:
            final_answer = response.text
            break

        contents.append(response.candidates[0].content)

        function_response_parts = []
        for fc in response.function_calls:
            args = dict(fc.args) if fc.args else {}
            reasoning.append({"tool": fc.name, "detail": f"Called {fc.name} with {args}"})

            fn = tool_functions.get(fc.name)
            try:
                result = fn(**args) if fn else f"Unknown tool: {fc.name}"
            except Exception as e:
                result = f"Error running {fc.name}: {e}"

            function_response_parts.append(
                types.Part.from_function_response(name=fc.name, response={"result": result})
            )

        contents.append(types.Content(role="tool", parts=function_response_parts))

    return {
        "answer": final_answer,
        "sources": sources,
        "reasoning": reasoning,
        "retrieved_contexts": retrieved_contexts,
    }

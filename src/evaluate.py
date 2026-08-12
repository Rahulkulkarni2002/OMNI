import json
import os
import time
import csv
from dotenv import load_dotenv
from google import genai
from google.genai import types
from agent import run_agent

load_dotenv()

judge_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=45000),
)

def with_retry(fn, *args, attempts=3, delay=8, **kwargs):
    last_error = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs), None
        except Exception as e:
            last_error = e
            if i < attempts - 1:
                print(f"      (attempt {i+1} failed: {e} -- retrying in {delay}s)")
                time.sleep(delay)
    return None, last_error


print("Loading test questions...")
with open("eval_questions.json", "r") as f:
    questions = json.load(f)

print(f"Running {len(questions)} questions through the live agent (with retries)...\n")
samples_data = []
for q in questions:
    print(f"  [{q['id']}] {q['question'][:60]}...")
    result, error = with_retry(run_agent, q["question"])
    if result is not None:
        samples_data.append({
            "id": q["id"], "category": q["category"], "question": q["question"],
            "answer": result["answer"], "agent_ok": True,
            "retrieved_contexts": result["retrieved_contexts"] or ["No context retrieved."],
        })
        print(f"      -> OK ({len(result['retrieved_contexts'])} context pieces)")
    else:
        samples_data.append({
            "id": q["id"], "category": q["category"], "question": q["question"],
            "answer": None, "agent_ok": False, "retrieved_contexts": [],
        })
        print(f"      -> FAILED after retries: {error}")
    time.sleep(3)

with open("eval_raw_results.json", "w") as f:
    json.dump(samples_data, f, indent=2)

print("\nScoring faithfulness with Gemini as judge (successful answers only)...\n")

def score_faithfulness(question, answer, contexts):
    context_text = "\n\n---\n\n".join(contexts)
    prompt = f"""You are evaluating whether an AI answer is FAITHFUL to its retrieved context (not hallucinated).

CONTEXT:
{context_text}

QUESTION: {question}
ANSWER: {answer}

Respond with ONLY a number between 0.0 and 1.0:
1.0 = fully supported, 0.5 = partially supported, 0.0 = unsupported or contradicts context.
Respond with ONLY the number."""
    response = judge_client.models.generate_content(model="gemini-3.1-flash-lite", contents=prompt)
    return float(response.text.strip())

results = []
for s in samples_data:
    if not s["agent_ok"]:
        print(f"  Skipping [{s['id']}] -- agent failed, nothing to score.")
        results.append({**s, "faithfulness": None})
        continue
    print(f"  Scoring [{s['id']}]...")
    score, error = with_retry(score_faithfulness, s["question"], s["answer"], s["retrieved_contexts"])
    print(f"      -> {score if score is not None else f'FAILED: {error}'}")
    results.append({**s, "faithfulness": score})
    time.sleep(3)

print("\n=== RESULTS ===")
for r in results:
    print(f"[{r['id']}] {r['category']:9s} agent_ok={r['agent_ok']!s:5s} faithfulness={r['faithfulness']}  {r['question'][:45]}")

agent_ok_count = sum(1 for r in results if r["agent_ok"])
print(f"\nAgent completion rate: {agent_ok_count}/{len(results)} ({agent_ok_count/len(results)*100:.0f}%)")

valid = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
if valid:
    print(f"Faithfulness (successfully scored): {sum(valid)/len(valid):.2f} ({len(valid)} questions)")

with open("eval_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "category", "question", "answer", "agent_ok", "faithfulness"])
    writer.writeheader()
    for r in results:
        writer.writerow({k: r.get(k) for k in ["id", "category", "question", "answer", "agent_ok", "faithfulness"]})
print("Saved to eval_results.csv")

import json
import time
from agent import run_agent

with open("eval_raw_results.json", "r") as f:
    samples_data = json.load(f)

failed = [s for s in samples_data if not s["agent_ok"]]
print(f"Retrying {len(failed)} previously failed questions, with more patience this time...\n")

for s in failed:
    print(f"  [{s['id']}] {s['question'][:60]}...")
    result = None
    for attempt in range(6):  # more attempts than before
        try:
            result = run_agent(s["question"])
            break
        except Exception as e:
            wait = 10 * (attempt + 1)  # 10s, 20s, 30s... growing backoff
            print(f"      (attempt {attempt+1} failed: {e} -- waiting {wait}s)")
            time.sleep(wait)

    if result is not None:
        s["answer"] = result["answer"]
        s["agent_ok"] = True
        s["retrieved_contexts"] = result["retrieved_contexts"] or ["No context retrieved."]
        print("      -> OK this time")
    else:
        print("      -> still failing after 6 attempts, leaving as-is")
    time.sleep(5)

with open("eval_raw_results.json", "w") as f:
    json.dump(samples_data, f, indent=2)

still_failing = sum(1 for s in samples_data if not s["agent_ok"])
print(f"\nDone. {len(samples_data) - still_failing}/{len(samples_data)} now succeeding.")

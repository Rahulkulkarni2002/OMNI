import json


def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += chunk_size - overlap
    return chunks


def chunk_all_docs(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    all_chunks = []
    for doc in docs:
        if not doc.get("content"):
            continue
        text_chunks = chunk_text(doc["content"])
        for i, chunk in enumerate(text_chunks):
            all_chunks.append(
                {
                    "title": doc["title"],
                    "url": doc["url"],
                    "chunk_id": f"{doc['title']}_{i}",
                    "text": chunk,
                }
            )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Created {len(all_chunks)} chunks from {len(docs)} docs")


if __name__ == "__main__":
    chunk_all_docs("../data/raw_docs/nim_docs.json", "../data/raw_docs/nim_chunks.json")

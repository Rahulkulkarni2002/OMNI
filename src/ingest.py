import asyncio
import aiohttp
import ssl
import certifi
import json
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://docs.nvidia.com/nim/large-language-models/latest/about-nim-llm/overview.html"

ssl_context = ssl.create_default_context(cafile=certifi.where())


async def fetch(session, url):
    async with session.get(url, ssl=ssl_context) as response:
        response.raise_for_status()
        return await response.text()


async def get_all_doc_links(session, start_url):
    html = await fetch(session, start_url)
    soup = BeautifulSoup(html, "html.parser")
    nav = soup.find("nav", class_="bd-docs-nav")
    links = nav.find_all("a")

    doc_links = []
    for link in links:
        href = link.get("href")
        text = link.get_text(strip=True)
        if href and href != "#":
            full_url = urljoin(start_url, href)
            doc_links.append({"title": text, "url": full_url})
    return doc_links


def extract_article_text(html):
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article", class_="bd-article")
    if not article:
        return None
    return article.get_text(separator="\n", strip=True)


async def fetch_and_extract(session, doc):
    try:
        html = await fetch(session, doc["url"])
        text = extract_article_text(html)
        return {"title": doc["title"], "url": doc["url"], "content": text}
    except Exception as e:
        print(f"Failed: {doc['url']} -> {e}")
        return None


async def main():
    async with aiohttp.ClientSession() as session:
        # Step 1: get all links (what you already have working)
        doc_links = await get_all_doc_links(session, BASE_URL)
        print(f"Found {len(doc_links)} pages\n")

        # Step 2: fetch + extract ALL pages concurrently
        tasks = [fetch_and_extract(session, doc) for doc in doc_links]
        results = await asyncio.gather(*tasks)

        # Step 3: drop any that failed (None values)
        dataset = [r for r in results if r is not None]

        # Step 4: save to JSON
        os.makedirs("../data/raw_docs", exist_ok=True)
        with open("../data/raw_docs/nim_docs.json", "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(dataset)} pages to nim_docs.json")


if __name__ == "__main__":
    asyncio.run(main())

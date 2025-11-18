import aiohttp
import asyncio
from bs4 import BeautifulSoup
from typing import List, Optional
from langchain_core.documents import Document

async def fetch_html(session: aiohttp.ClientSession, url: str) -> str:
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with session.get(url, timeout=timeout) as response:
            response.raise_for_status()
            html = await response.text()
            return html
    except Exception as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return ""

def parse_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Remove scripts and styles
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    return soup.get_text(separator="\n").strip()

async def scrape_single_url(url: str) -> Optional[Document]:
    async with aiohttp.ClientSession() as session:
        html = await fetch_html(session, url)
        text = parse_html_to_text(html)
        if text:
            return Document(page_content=text, metadata={"source": url})
        return None

async def scrape_urls(urls: List[str]) -> List[Document]:
    tasks = [scrape_single_url(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return [doc for doc in results if doc is not None]

# Helper for sync code to call async
def scrape_urls_sync(urls: List[str]) -> List[Document]:
    return asyncio.run(scrape_urls(urls))
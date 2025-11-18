from typing import List, Optional
from langchain_core.documents import Document
from .web_scraper import scrape_urls_sync

# Web links to scrape
WEB_SOURCES = [
    # NDRRMC
    "https://ndrrmc.gov.ph/",
    "https://monitoring-dashboard.ndrrmc.gov.ph/page/rainfall",
    "https://monitoring-dashboard.ndrrmc.gov.ph/page/rainfall_warning",
    "https://monitoring-dashboard.ndrrmc.gov.ph/page/weather_advisory",
    "https://monitoring-dashboard.ndrrmc.gov.ph/page/weather"
    # TODO: Add more relevant disaster-related sources
]

def fetch_disaster_data(pdf_chunks: Optional[List[Document]] = None) -> List[Document]:
    all_docs = []
    
    # Scrape web sources
    try:
        print("[DEBUG] Scraping web sources...")
        web_docs = scrape_urls_sync(WEB_SOURCES)
        print(f"[DEBUG] Scraped {len(web_docs)} web documents")
        
        if web_docs:
            all_docs.extend(web_docs)
    except Exception as e:
        print(f"[ERROR] Web scraping failed: {e}")
    
    # PDF fallback
    if pdf_chunks:
        print(f"[DEBUG] Adding {len(pdf_chunks)} PDF fallback chunks...")
        all_docs.extend(pdf_chunks)
    
    if not all_docs:
        print("[WARNING] No documents were collected from any source!")
    else:
        print(f"[DEBUG] Total documents collected: {len(all_docs)}")
    
    return all_docs
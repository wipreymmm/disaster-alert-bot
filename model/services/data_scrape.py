from .web_scraper import scrape_urls_sync
from langchain_core.documents import Document
from typing import List, Optional

# Web links to scrape
WEB_SOURCES = [
    # Typhoon Preparedness
    "https://www.ready.gov/tl/kids/disaster-facts/hurricanes",
    # "https://www.unicef.org/vietnam/typhoon-safety-information-parents",
    # "https://www.unicef.org/parenting/emergencies/how-to-prepare-for-hurricane-or-typhoon",
    "https://www.ecoflow.com/ph/blog/what-to-do-before-during-and-after-a-typhoon",

    # Flood Preparedness
    "https://www.pagasa.dost.gov.ph/learning-tools/floods",
    "https://www.ready.gov/floods",
    "https://www.ready.gov/tl/floods",

    # Tornado Preparedness
    "https://www.ready.gov/tl/kids/disaster-facts/tornadoes",

    # Earthquake Preparedness
    "https://www.ecoflow.com/ph/blog/what-to-do-before-during-and-after-an-earthquake",
    "https://www.ready.gov/tl/earthquakes",

    # Landslide Preparedness
    "https://www.ecoflow.com/ph/blog/how-to-prevent-landslides",
    #"https://www.suyoilocossur.gov.ph/reminders-what-to-do-before-during-and-after-landslides/",

    # Tsunami Preparedness
    "https://www.phivolcs.dost.gov.ph/introduction-to-tsunami/",
    "https://www.ready.gov/tsunamis",
    "https://www.ready.gov/tl/tsunamis",
    #"https://uwiseismic.com/tsunamis/preparedness/",

    # Volcanic Eruption Preparedness
    #"https://www.unicef.org/philippines/emergency-preparedness-tips-volcanic-eruptions",
    "https://www.ready.gov/tl/volcanoes",
    "https://www.ready.gov/volcanoes",

    # Man Made Disasters
    "https://ltoportal.ph/what-to-do-car-accident-philippines/",
    "https://www.ready.gov/explosions",
    "https://www.ready.gov/tl/explosion",
    "https://www.ready.gov/home-fires",
    "https://www.ready.gov/tl/home-fires",
    "https://houseofit.ph/how-to-protect-against-cyber-threats-in-the-philippines/",
    
    # TODO: Add more sources 
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
        else:
            print("[WARNING] No web documents were scraped successfully")
            
    except Exception as e:
        print(f"[ERROR] Web scraping failed: {e}")

    if not all_docs: 
        print("[WARNING] No documents were collected from web sources!")
    else: 
        print(f"[DEBUG] Total web documents collected: {len(all_docs)}")
    
    return all_docs
import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)


load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", show_progress_bar=False, chunk_size=50,
    retry_min_seconds=10
)

chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

async def main():
    """Main sync function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "TavilyCrawl: Starting to Crawl documentation from "
        "https://python.langchain.com/",
        Colors.PURPLE,
    )

    #Crawl the documentation site
    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 5,
        "extra_depth": "advanced",
    })
    all_docs = [Document(page_content=result['raw_content'],
                         metadata={"source": result['url']}) for result in res['results']]
    log_success(
        f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documents site.",
    )

if __name__ == "__main__":
    asyncio.run(main())
import asyncio
import os
import ssl
from typing import List

import certifi
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OllamaEmbeddings(model="nomic-embed-text")


vectorstore = PineconeVectorStore(
    index_name=os.environ["PINECONE_INDEX_NAME"], embedding=embeddings
)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """process documents in batches asynchronously"""
    log_header("Vector Storage Phase")
    log_info(
        f"vectorstore indexing preparing to add {len(documents)} documents into vectorstore",
        Colors.DARKCYAN,
    )

    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]
    log_info(
        f"Documents is split into {len(batches)} batches of {len(batches)} documents each"
    )

    async def add_batch(batch: List[Document], batch_num: int):

        try:
            await vectorstore.aadd_documents(batch)

            log_success(
                f"Vectorstore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"Vectorstore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # count successfull batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"Vectorstore Indexing: All batches processed successfully ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"Vectorstore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    """Main async function to orchestrate the entire process"""
    log_header("Documentation INGESTION PIPELINE")

    log_info(
        "TavilyCrawl: Starting to Crawl documentations from https://python.langchain.com/",
        Colors.PURPLE,
    )

    results = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 5,
            "extract_depth": "advanced",
        }
    )

    # convert to docs
    all_docs = [
        Document(page_content=result["raw_content"], metadata={"source": result["url"]})
        for result in results["results"]
    ]

    # split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_documents = text_splitter.split_documents(all_docs)
    log_success(
        f"Texsplitter created {len(splitted_documents)} chunks from {len(all_docs)} documents."
    )

    # embedding and store to vector db
    await index_documents_async(splitted_documents, batch_size=500)

    log_header("PIPELINE COMPLETED")
    log_success("Documentation ingestion pipeline finished successfully")
    log_info("Summary", Colors.BOLD)
    # log_info(f" = URLs mapped: {len(site_map['results'])}")
    log_info(f" = Documents extracted: {len(all_docs)}")
    log_info(f" = Chunks created: {len(splitted_documents)}")


if __name__ == "__main__":
    asyncio.run(main())

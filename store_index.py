from dotenv import load_dotenv
import os

from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

extracted_data = load_pdf_files("data/")
filter_data = filter_to_minimal_docs(extracted_data)
texts_chunk = text_split(filter_data)

embeddings = download_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# connect to index
index = pc.Index(index_name)

# store vectors
docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embeddings,
    index_name=index_name
)

print("✅ Pinecone index updated successfully!")

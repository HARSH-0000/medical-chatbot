from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

print("Loading PDF Data...")
extracted_data = load_pdf_file("data/")
print("Splitting text...")
text_chunks = text_split(extracted_data)
print("Downloading Embeddings...")
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"

print("Uploading to Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)
print("Completed successfully!")

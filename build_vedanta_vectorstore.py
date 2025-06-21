import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from load_pdfs_as_documents import load_pdfs_as_documents
from dotenv import load_dotenv
pdf_dir="./vedanta_hin_san_texts"
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def build_vectorstore(pdf_dir, vectorstore_dir="vedanta_vectorstore"):
    docs = load_pdfs_as_documents(pdf_dir)
    print(f"Loaded {len(docs)} page documents.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(chunks, embeddings)

    vectordb.save_local(vectorstore_dir)
    print(f"✅ FAISS vectorstore saved to `{vectorstore_dir}`")

if __name__ == "__main__":
    build_vectorstore(pdf_dir)

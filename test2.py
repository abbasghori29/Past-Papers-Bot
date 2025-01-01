from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders.mongodb import MongodbLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import FAISS
from pymongo import MongoClient
import numpy as np
from bson import ObjectId
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
import logging
from langchain_community.document_loaders import DirectoryLoader


# Set up logging
os.environ["GOOGLE_API_KEY"] = "AIzaSyBKWYpNkHveC8p_pu94SNmy8GByapkDlE0"

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
directory_path = '2016 November Examination Session'
loader = DirectoryLoader(directory_path, recursive=True)

# Load documents from the directory
documents = loader.load()

# Create a FAISS vector store with the documents and embeddings
vector_store = FAISS.from_documents(documents, embeddings_model)

# Function to search for documents based on a user query
def search_documents(query, k=5):
    # Perform the search using similarity search
    results = vector_store.similarity_search(query, k)
    # Extract and return the paths of the documents
    return [result.metadata['source'] for result in results]

# Example usage: search for documents based on a query
user_query = "english"
document_paths = search_documents(user_query)

# Print the paths of the documents that match the query
print(document_paths)
from src.logger.logging import logging
from src.exceptions.exception import customexception
import sys
import faiss
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, OllamaEmbeddings
import numpy as np
from src.components.document_loader import DocumentLoader
from src.components.text_splitter import TextSplitter


class Embedding:
    model_name = "BAAI/bge-small-en"
    hf_embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    ollama_embeddings = OllamaEmbeddings(model="llama3.1")
    
    
class VectorStore:
    def __init__(self):
        self.embedding= Embedding()
        
    def initiate_vector_store(self, splits):
        logging.info("Vector store has been started")
        try:
            texts = [doc.page_content for doc in splits]
            
            # for Ollama embeddings use this line of code
            # embeddings = np.array(self.embedding.ollama_embeddings.embed_documents(texts))
            
            # for hf_embeddings use this line of code
            embeddings = np.array(self.embedding.hf_embeddings.embed_documents(texts))
        
            dimension = 384  # Dimensionality of embeddings
            num_neighbors = 32  # Number of neighbors for the HNSW graph
            
            # Create HNSW index
            index_hnsw = faiss.IndexHNSWFlat(dimension, num_neighbors)
            index_hnsw.hnsw.efConstruction = 200  # Parameter controlling graph construction quality
            
            # Add embeddings to the index
            index_hnsw.add(embeddings)
            
            faiss.write_index(index_hnsw, "./hnsw_index.faiss")
            
            print("Index saved successfully!")
        except Exception as e:
            logging.info("Exception occured in Vector store part")
            raise customexception(e, sys)
        
        
document_loader = DocumentLoader()
documents = document_loader.initiate_document_loader()
text_splitter = TextSplitter()
splits = text_splitter.initiate_text_splitter(documents)
vector_store = VectorStore()
vector_store.initiate_vector_store(splits)
    
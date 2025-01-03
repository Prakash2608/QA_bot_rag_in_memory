from src.logger.logging import logging
from src.exceptions.exception import customexception
import sys
import faiss
import numpy as np
# from src.components.vectorstore import Embedding
import json
import joblib


class Retriever:
    def __init__(self):
        # self.embeddings = Embedding()
        pass
    
    def retrieve_context(self, query):
        try:
            logging.info("Retrieving context")
            # To load the model
            embedding_model = joblib.load('hf_embedding_model.pkl')
            query_vector = embedding_model.embed_documents([query])[0]
            
            # Load the index from the saved file
            loaded_index = faiss.read_index("./hnsw_index.faiss")
            
            # Load texts
            with open("texts.json", "r") as f:
                texts = json.load(f)
                
            # Perform the search
            k = 5  # Number of nearest neighbors
            distances, indices = loaded_index.search(np.array([query_vector]), k)
            
            # Retrieve the documents corresponding to the closest embeddings
            retrieved_docs = [texts[i] for i in indices[0]]

            return query, retrieved_docs
        except Exception as e:
            logging.info("Exception occured in Retriever part")
            raise customexception(e, sys)
        
retriever = Retriever()
retrieved_docs = retriever.retrieve_context("Explain me the attention mechanism in easy methods")
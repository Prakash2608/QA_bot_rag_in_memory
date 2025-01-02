from src.logger.logging import logging
from src.exceptions.exception import customexception
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextSplitter:
    def __init__(self):
        pass
    
    def initiate_text_splitter(self, docs):
        try:
            logging.info("Text splitter has started.")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            return splits
            
        except Exception as e:
            logging.info("Exception occured in document loader part")
            raise customexception(e, sys)

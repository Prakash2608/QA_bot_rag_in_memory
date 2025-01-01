from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
import bs4
from src.logger.logging import logging
from src.exceptions.exception import customexception
import sys
import yaml
import os
# from root import from_root


class PdfLoaderConfig:
    pdf_directory = 'C:/Users/praka/unique_projects/QA_bot_rag_in_memory/PDF'
    
    # List all files in the directory
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    
    # # Load each PDF
    # pdf_documents = []
    # for pdf_file in pdf_files:
    #     pdf_path = os.path.join(pdf_directory, pdf_file)
    #     loader = PyPDFLoader(pdf_path)
    #     pdf_documents.extend(loader.load())
    
    
class WebLoaderConfig:
    # Load the YAML file
    with open('C:/Users/praka/unique_projects/QA_bot_rag_in_memory/web_paths.yaml', 'r') as file:
        data = yaml.safe_load(file)
    
    web_paths = data.get('web_paths', [])
    
class TextLoaderConfig:
    
    # Path to the folder containing your text files
    text_folder_path = "C:/Users/praka/unique_projects/QA_bot_rag_in_memory/text_files"
    
    # List all text files in the directory
    text_files = [f for f in os.listdir(text_folder_path) if f.endswith('.txt')]
    
    # Load documents using TextLoader
    # documents = []
    # for text_file in text_files:
    #     file_path = os.path.join(text_folder_path, text_file)
    #     loader = TextLoader(file_path)
    #     documents.extend(loader.load())
    
class DocumentLoader:
    def __init__(self):
        self.text_loader_config = TextLoaderConfig()
        self.web_loader_config = WebLoaderConfig()
        self.pdf_loader_config = PdfLoaderConfig()
        
        
    def initiate_document_loader(self):
        logging.info("Document Loader started")
        try:
            web_loader = WebBaseLoader(
                web_paths=self.web_loader_config.web_paths,
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(class_=("post-content", "post_title", "post-header"))
                ),
            )
            
            web_docs = web_loader.load()
            
            # Load each PDF
            pdf_documents = []
            for pdf_file in self.pdf_loader_config.pdf_files:
                pdf_path = os.path.join(self.pdf_loader_config.pdf_directory, pdf_file)
                loader = PyPDFLoader(pdf_path)
                pdf_documents.extend(loader.load())
            
            # Load documents using TextLoader
            text_documents = []
            for text_file in self.text_loader_config.text_files:
                file_path = os.path.join(self.text_loader_config.text_folder_path, text_file)
                loader = TextLoader(file_path)
                text_documents.extend(loader.load())
                
            documents = web_docs + pdf_documents + text_documents
            # print(len(documents))
            return documents
            
        except Exception as e:
            logging.info("Exception occured in document loader part")
            raise customexception(e, sys)
         
         
document_loader = DocumentLoader()
document_loader.initiate_document_loader()
    
    

    



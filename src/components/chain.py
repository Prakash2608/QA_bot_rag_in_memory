from src.logger.logging import logging
from src.exceptions.exception import customexception
import sys
import os 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from src.components.retrieval import Retriever

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")


from langchain.schema import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import OllamaEmbeddings

class Prompt:
    @staticmethod
    def generate_prompt(context, question):
        return f"""
        You are to respond with intelligence, politeness, and kindness.
        Always maintain a calm and respectful tone. Avoid any abusive, violent, or offensive language.
        Strive to provide clear, thoughtful, and empathetic responses to all queries while fostering a positive interaction.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """

class Chain:
    def __init__(self):
        self.prompt = Prompt()
        self.llm = ChatGroq(model="llama3-8b-8192", temperature=0.1)
        
    def initiate_chain(self, query, retrieved_documents):
        try:
            logging.info("Initiating the Chain")
            context = retrieved_documents
            
            # Generate the formatted prompt
            formatted_prompt = self.prompt.generate_prompt(context, query)
            
            # Wrap the prompt in a HumanMessage
            message = HumanMessage(content=formatted_prompt)
            
            # Use invoke instead of __call__
            response = self.llm.invoke([message])
            
            # Parse the response
            parsed_response = StrOutputParser().parse(response.content)
            
            return parsed_response
        except Exception as e:
            logging.info("Exception occurred in chain part")
            raise customexception(e, sys)


        
retriever = Retriever()
query, retrieved_docs = retriever.retrieve_context("Can you explain me query vector, key vector and value vector in attention mechanism like a I am 5 year old boy.")
chain = Chain()
answer = chain.initiate_chain(query, retrieved_docs)
print(answer)

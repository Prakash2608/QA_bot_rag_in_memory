from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'



def get_requirements(file_path:str) -> List[str]:
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        
        if HYPEN_E_DOT in requirements: 
            requirements.remove(HYPEN_E_DOT)
    
    return requirements


setup(
    name="qasystem",
    version="0.0.1",
    author="PrakashR",
    author_email="prakashraj822@gmail.com",
    packages=find_packages(),
#     install_requires=["langchain","langchainhub","bs4","tiktoken","openai","boto3==1.34.37","langchain_community","chromadb","awscli",
# "streamlit",
# "pypdf",
# "faiss-cpu"]
    install_requires = get_requirements("./requirements.txt")
)
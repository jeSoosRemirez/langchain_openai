import os
from typing import List

import openai
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever

# .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
FILE_NAME = os.getenv("FILE_NAME")

# Langchain configs to be used
llm = OpenAI(model_name="text-davinci-003")
chat = ChatOpenAI(model_name="gpt-3.5-turbo")

# A context or rules for chat
template = """
        Present yourself as "NiftyBridge AI assistant", don't answer questions that doesn't concerns
        program "Nifty Bridge". Look for answers only in "vectorstore" document. If you don't know
        the answer then say "I don't know please contact with support by email support@nifty-bridge.com".
        Answer the question: {question}
        """


def get_loader(data_path: str = FILE_NAME) -> List[Document]:
    """
    Loads PDF file, returns loaded document
    """
    loader = PyPDFLoader(data_path)
    documents = loader.load()

    return documents


def vector_store(document: List[Document]) -> VectorStoreRetriever:
    """
    Returns VectorStoreRetriever from Chroma,
    used for RetrievalQA in qa_prompt()
    """

    # Splitting into chunks
    text_spliter = CharacterTextSplitter(
        separator=" ",
        chunk_size=1000,
        chunk_overlap=0,
    )
    texts = text_spliter.split_documents(document)

    # Creating embeddings and storing as vectors in Chroma
    embedding = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embedding)
    retriever = db.as_retriever(search_type="similarity")

    return retriever


def qa_prompt(retriever: VectorStoreRetriever, question: str = "Hi!") -> dict:
    """
    Returns response of chat on the user question
    return: 'resp': contains 'query'-template, 'result'-answer of chat, 'source_documents'-docs that been used
    """

    # Create prompt
    prompt = PromptTemplate(
        template=template, input_variables=["question"], validate_template=False
    )
    prompt = prompt.format(question=question)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        # chain_type_kwargs={"prompt": prompt}  A common issue of the lib occurs, not fixed yet
    )

    # Asking a question to chats
    resp = qa({"query": prompt})

    return resp

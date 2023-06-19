import os
from dotenv import load_dotenv
import openai
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


# .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
DATA_PATH = os.getenv("DATA_PATH")

# Langchain configs to be used
# llm = OpenAI(model_name="text-davinci-003")
# chat = ChatOpenAI(model_name="gpt-3.5-turbo")

template = """
        Present yourself as "NiftyBridge AI assistant", don't answer questions that doesn't concerns
        program "Nifty Bridge". Look for answers only in "vectorstore" document. If you don't know
        the answer then say "I don't know please contact with support by email support@nifty-bridge.com".
        Answer the question: {question}
        """


class OpenAIModelSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = OpenAI(model_name="text-davinci-003")
        return cls._instance


class ChatOpenAISingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = ChatOpenAI(model_name="gpt-3.5-turbo")
        return cls._instance


class PyPDFLoaderSingleton:
    _instance = None

    @classmethod
    def get_instance(cls, data_path):
        if not cls._instance:
            cls._instance = PyPDFLoader(data_path)
        return cls._instance


llm = OpenAIModelSingleton.get_instance()
chat = ChatOpenAISingleton.get_instance()


def get_loader(data_path: str = DATA_PATH):
    """
    Loads PDF file, returns loaded document
    """
    loader = PyPDFLoader(data_path)
    document = loader.load()

    return document


def vector_store(document):
    """
    Returns VectorStoreRetriever from Chroma,
    used for RetrievalQA in qa_prompt()

    inputs: 'document'-get_loader() return,
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


async def qa_prompt(retriever, question: str="Hi!"):
    """
    Returns response of chat on the user question

    inputs: 'question'-the input of the user to be used by chat, 'retriever'-vector_store() return
    return: 'resp': contains 'query'-template, 'result'-answer of chat, 'source_documents'-docs that been used
    'template': a context or rules for chat
    """

    # Create prompt
    prompt = PromptTemplate(
        template=template,
        input_variables=["question"],
        validate_template=False
    )
    prompt = prompt.format(question=question)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        # chain_type_kwargs={"prompt": prompt}  A common issue of the lib occurs, not fixed yet
    )

    # Asking a question to chat
    resp = await qa({"query": prompt})

    return resp

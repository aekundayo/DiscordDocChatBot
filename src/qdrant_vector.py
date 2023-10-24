from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS, Qdrant, weaviate, Redis
import qdrant_client    
import os

vectorpath = 'docs/vectorstore'
url = os.getenv("QDRANT_HOST_STRING"),


def get_Qvector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    
    memory_vectorstore = Qdrant.from_documents(
    text_chunks,
    embeddings,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="current_document",
    force_recreate=True,
    )


    client = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_HOST_STRING")
    )
    qdrant = Qdrant(
    client=client,
    embeddings=embeddings,
    collection_name="my_documents"
    )
    qdrant.add_texts(text_chunks)


    return memory_vectorstore 

def get_Qvector_store_from_docs(documents):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    
    memory_vectorstore = Qdrant.from_documents(
    documents,
    embeddings,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="current_document",
    force_recreate=True,
    )


    client = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_HOST_STRING")
    )
    qdrant = Qdrant(
    client=client,
    embeddings=embeddings,
    collection_name="my_documents"
    )
    qdrant.add_documents(documents)


    return memory_vectorstore 


    return memory_vectorstore 


def return_qdrant():    
    embeddings = OpenAIEmbeddings()
    client = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_HOST_STRING")
    )
    qdrant = Qdrant(
    client=client,
    embeddings=embeddings,
    collection_name="my_documents"
    )

    return qdrant

def update_qdrant(text_chunks):
    embeddings = OpenAIEmbeddings()
    client = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_HOST_STRING")
    )
    qdrant = Qdrant(
    client=client,
    embeddings=embeddings,
    collection_name="my_documents"
    )
    qdrant.add_texts(text_chunks)



def is_folder_empty(folder_path):
    return len(os.listdir(folder_path)) == 0




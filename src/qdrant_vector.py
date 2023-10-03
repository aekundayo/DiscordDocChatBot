from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS, Qdrant, weaviate, Redis
import qdrant_client    
import os

vectorpath = './docs/vectorstore'
url = os.getenv("QDRANT_HOST_STRING"),
def get_Qvector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    
    memory_vectorstore = Qdrant.from_texts(
    text_chunks,
    embeddings,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="my_documents",
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

def is_folder_empty(folder_path):
    return len(os.listdir(folder_path)) == 0




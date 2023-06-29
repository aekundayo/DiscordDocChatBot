import openai
import discord
from discord.ext import commands
import asyncio
import os
import time
import requests
import re
import json
import logging
import wandb
import random
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from wandb.integration.langchain import WandbTracer

# Set up logging
logging.basicConfig(level=logging.INFO)
openai.api_key = os.getenv('OPENAI_API_KEY')
scraperapi_key = os.getenv('SCRAPER_API_KEY')
brave_search_api_key = os.getenv('BRAVE_SEARCH_API_KEY')
intents = discord.Intents.all()
bot = commands.Bot("/", intents=intents)
vectorstore=None
load_dotenv()
def get_pdf_text(path):
    # define the path
    
    for filename in os.listdir(path):
    # check if the file is a pdf
      if filename.endswith('.pdf'):
        text = ""
        with open(os.path.join(path, filename), 'rb') as pdf_doc:
        
            pdf_reader = PdfReader(pdf_doc)
            for page in pdf_reader.pages:
                text += page.extract_text()
            
        return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="tiiuae/falcon-40b", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

path='./testdocs'
filename='finetune-medical.pdf'

raw_text = get_pdf_text(path)
chunks = get_text_chunks(raw_text)
vectorstore = get_vectorstore(chunks)

llm = HuggingFaceHub(repo_id="tiiuae/falcon-40b-instruct", task="summarization", model_kwargs={"temperature":0.7, "max_length":512})
    #llm = HuggingFacePipeline.from_model_id(model_id="tiiuae/falcon-40b-instruct", task="summarization", model_kwargs={"temperature":0, "max_length":64})
logging.info("SETTING MODEL TO HUGGING FACE")


wandb_config = {"project": "wandb_prompts_quickstart"}
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(),callbacks=[WandbTracer(wandb_config)])
query = "Give and extremely detailed Summary of this document, including a title and ALL the important ideas expressed in the document as bullet points in markdown format"
answer=qa.run(query)
logging.info(answer)
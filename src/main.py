import openai
import anthropic
import discord
from discord.ext import commands
import asyncio
import os
import time
import logging
import wandb
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.llms import HuggingFaceHub,OpenAI
from langchain import HuggingFacePipeline
from wandb.integration.langchain import WandbTracer
from langchain.document_loaders import BSHTMLLoader
from summary_prompts import get_guidelines
from utils import extract_url, download_html, get_pdf_text, get_text_chunks, create_directories_if_not_exists, extract_yt_transcript, extract_text_from_htmls, unzip_website
from vector import get_vectorstore, get_history_vectorstore, persist_new_chunks

from concurrent.futures import ThreadPoolExecutor

#constants
vectorpath = './docs/vectorstore'
pdf_path = './docs/pdfs'
web_doc_path = './docs/web'
zip_path = './docs/zip'

# Set up logging
logging.basicConfig(level=logging.INFO)
#GET API KEYS FROM .ENV FILE
#claude.api_key = os.getenv('ANTHROPIC_API_KEY')
openai.api_key = os.getenv('OPENAI_API_KEY')
scraperapi_key = os.getenv('SCRAPER_API_KEY')
brave_search_api_key = os.getenv('BRAVE_SEARCH_API_KEY')
intents = discord.Intents.all()
bot = commands.Bot("/", intents=intents)

# Global Variables
vectorstore=None
dialog_contexts = {}

class DialogContext:
#load dot env file
  load_dotenv()
  if os.getenv('DEV_MODE'):
    wandb.login
    wandb_config = {"project": "DiscordChatBot"}
  def __init__(self, maxlen=5):
    self.maxlen = maxlen
    self.history = []
  
  #add message to history
  #role: user or bot
  def add_message(self, role, content):
    if len(self.history) >= self.maxlen:
      self.history.pop(0)
    self.history.append({"role": role, "content": content})
  
  def get_messages(self):
    return self.history.copy()






def get_conversation_chain(vectorstore):
    llm = OpenAI()
    #llm = HuggingFaceHub(repo_id="tiiuae/falcon-40b", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=return_llm(),
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


  
def get_current_date():
  return str(time.strftime("%Y-%m-%d, %H:%M:%S"))


def openAIGPTCall(messages, model="gpt-4", temperature=0):
  start_time = time.time()
  if os.getenv('DEV_MODE'):
    wandb_config = {"project": "wandb_prompts_quickstart"}
    response = openai.ChatCompletion.create(model=model,
                                            messages=messages,
                                            temperature=temperature,callbacks=[WandbTracer(wandb_config)])
  else:
    response = openai.ChatCompletion.create(model=model,
                                            messages=messages,
                                            temperature=temperature)
  elapsed_time = (time.time() - start_time) * 1000
  cost_factor = 0.04
  cost = cost_factor * (response.usage["total_tokens"] / 1000)
  message = response.choices[0].message.content.strip()
  return message, cost, elapsed_time


async def send_long_message(channel, message):
  max_length = 2000
  chunks = [
    message[i:i + max_length] for i in range(0, len(message), max_length)
  ]
  for chunk in chunks:
    await channel.send(chunk)

def return_llm(provider="openai"):
  if provider == "openai":
    if os.getenv('DEV_MODE'):
      wandb_config = {"project": "wandb_prompts_quickstart"}
      return ChatOpenAI(model_name="gpt-3.5-turbo-16k",callbacks=[WandbTracer(wandb_config)])
    else:
      return ChatOpenAI(model_name="gpt-3.5-turbo-16k")
  elif provider == "huggingface":
    return HuggingFaceHub(repo_id="tiiuae/falcon-40b", model_kwargs={"temperature":0.5, "max_length":512})
  elif provider == "anthropic":
    return ChatAnthropic(model_name="claude-2")

def retrieve_answer(vectorstore):
  llm = return_llm()  
  
  #if message.content.startswith('!hf'): 
  #  llm = HuggingFaceHub(repo_id="tiiuae/falcon-40b", task="summarization", model_kwargs={"temperature":0.5, "max_length":1512})
  #  #llm = HuggingFacePipeline.from_model_id(model_id="tiiuae/falcon-40b-instruct", task="summarization", model_kwargs={"temperature":0, "max_length":64})
  #  logging.info("SETTING MODEL TO HUGGING FACE")
    
  if os.getenv('DEV_MODE'):
    wandb_config = {"project": "wandb_prompts_quickstart"}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(),callbacks=[WandbTracer(wandb_config)])
  else:
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
  guidelines = get_guidelines()
  query = guidelines[2]
  #query = "You are a helpful assistant with concise and accurate responses given in the tone of a professional presentation. Give and detailed Summary of this document making sure to include the following sections Title: Authors: Ideas: Conclusions:"
  answer=qa.run(query)
  logging.info(answer)
  return answer

async def get_message_history(channel):
  messages = []
  async for msg in channel.history(limit=5):
      messages.append(msg)

  msg_history = ""
  for msg in messages:
      msg_history += f"{msg.author.name}: {msg.content}\n"
  return msg_history
@bot.event
async def on_ready():
  logging.info(f'{bot.user} has connected to Discord!')

#Event handler for bot messages. Could break into smaller functions.
@bot.event
async def on_message(message):
  if message.author == bot.user:
    return
  
  if os.getenv('DEV_MODE'):
    global wandb_config
    wandb_config = {"project": "wandb_prompts_quickstart"}
  logging.info(f"MESSAGE RECEIVED!")
  
  if message.author == bot.user:
    return
  #Process PDFs 


  create_directories_if_not_exists(pdf_path)
  web_doc = web_doc_path +'/download.html'
  create_directories_if_not_exists('./docs/web')
  create_directories_if_not_exists(zip_path)
  create_directories_if_not_exists(vectorpath)







  if 'http://' in  message.content or 'https://' in message.content:
    chunks = None
    if 'youtube.com' in message.content or 'youtu.be' in message.content:
        raw_text = extract_yt_transcript(message.content)
        chunks = get_text_chunks(''.join(raw_text))
        vectorstore = get_vectorstore(chunks)
        executor = ThreadPoolExecutor()
        future = executor.submit(persist_new_chunks, chunks)
        answer = retrieve_answer(vectorstore)
        await send_long_message(message.channel, answer)
    else:        
        urls = extract_url(message.content)   
        for url in urls:
            download_html(url, web_doc)
            loader = BSHTMLLoader(web_doc)
            data = loader.load()
        
            for page_info in data:
                chunks = get_text_chunks(page_info.page_content)
                vectorstore = get_vectorstore(chunks)
                executor = ThreadPoolExecutor()
                future = executor.submit(persist_new_chunks, chunks)
                answer = retrieve_answer(vectorstore=vectorstore)
                os.remove(web_doc) # Change here
                await send_long_message(message.channel, answer)

#handle Attachments (pdf and zip)
  if message.attachments:
    for attachment in message.attachments:
        if attachment.filename.endswith('.pdf'):  # if the attachment is a pdf
          data = await attachment.read()  # read the content of the file
          with open(os.path.join(pdf_path, attachment.filename), 'wb') as f:  # save the pdf to a file
              f.write(data)
          raw_text = get_pdf_text(pdf_path)
          chunks = get_text_chunks(raw_text)
          vectorstore = get_vectorstore(chunks)
          executor = ThreadPoolExecutor()
          future = executor.submit(persist_new_chunks, chunks)
          answer = retrieve_answer(vectorstore=vectorstore)
          await send_long_message(message.channel, answer)
          return
        elif attachment.filename.endswith('.zip'):  # if the attachment is a pdf
          data = await attachment.read()  # read the content of the file
          with open(os.path.join(zip_path, attachment.filename), 'wb') as f:  # save the pdf to a file
              f.write(data)
          unzip_website(zip_path+"/"+attachment.filename, zip_path)
          await extract_text_from_htmls(zip_path)
          return

          
  else:
    user_prompt = message.content
    logging.info(f"Received message from {message.author.name}: {user_prompt}")
    message_history = await get_message_history(message.channel)
    query = "You are a helpful assistant with concise and accurate responses given in the tone of a professional presentation. Try and answer the question as truthfully as possible. What is the answer to the question: " + user_prompt + message_history

    hitory_vectorstore = get_history_vectorstore()
    if os.getenv('DEV_MODE'):
      wandb_config = {"project": "wandb_prompts_quickstart"}
      qa = RetrievalQA.from_chain_type(llm=return_llm(), chain_type="stuff", retriever=hitory_vectorstore.as_retriever(), return_source_documents=True,callbacks=[WandbTracer(wandb_config)])
    else:
      qa = RetrievalQA.from_chain_type(llm=return_llm(), chain_type="stuff", retriever=hitory_vectorstore.as_retriever(), return_source_documents=True)
   
    query = "You are a helpful assistant with concise and accurate responses given in the tone of a professional presentation. Try and answer the question as truthfully as possible. What is the answer to the question: " + user_prompt
    answer=qa.run(query)
    logging.info(answer)

    await send_long_message(message.channel, answer)





bot.run(os.getenv('DISCORD_TOKEN'))
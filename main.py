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
import requests
import urllib.parse as urlparse
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS, qdrant, weaviate, Redis
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub,OpenAI
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredHTMLLoader
from wandb.integration.langchain import WandbTracer
from langchain.document_loaders import BSHTMLLoader
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi


from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO)
#GET API KEYS FROM .ENV FILE
openai.api_key = os.getenv('OPENAI_API_KEY')
scraperapi_key = os.getenv('SCRAPER_API_KEY')
brave_search_api_key = os.getenv('BRAVE_SEARCH_API_KEY')
intents = discord.Intents.all()
bot = commands.Bot("/", intents=intents)
vectorstore=None

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


dialog_contexts = {}


def extract_url(s):
    # Regular expression to match URLs
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    urls = re.findall(url_pattern, s)
    return urls


def download_html(url, filepath):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check that the GET request was successful
    if response.status_code == 200:
        # Write the response content (the HTML) to a file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)
    else:
        print(f'Failed to download {url}: status code {response.status_code}')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("THIS PAGE CANT BE DOWNLOADED: "+response.content.decode("utf-8"))


    
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
            os.remove(os.path.join(path, filename))
        return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_discord_docs")
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = OpenAI()
    #llm = HuggingFaceHub(repo_id="tiiuae/falcon-40b", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
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


def scrape_and_summarize(url):
  api_url = 'https://async.scraperapi.com/jobs'
  data = {"apiKey": scraperapi_key, "url": url}
  headers = {"Content-Type": "application/json"}
  response = requests.post(api_url, headers=headers, json=data)
  logging.info(f"ScraperAPI response: {response.text}")
  if response.status_code == 200:
    job = response.json()
    return job["id"]
  else:
    return None


async def check_job_status(job_id, channel):
  while True:
    await asyncio.sleep(10)  # check every 10 seconds
    status_url = f"https://async.scraperapi.com/jobs/{job_id}"
    response = requests.get(status_url)
    logging.info(f"ScraperAPI response: {response.text}")
    if response.status_code == 200:
      job = response.json()
      if job["status"] == "finished":
        # once the job is completed, we get the result
        result_url = f"https://async.scraperapi.com/jobs/{job_id}"
        result_response = requests.get(result_url)
        if result_response.text:
          try:
            result = result_response.json()
            system_prompt = "Please provide a brief summary of the following content: " + str(
              result)
            summary, _, _ = openAIGPTCall([{
              "role": "system",
              "content": system_prompt
            }])
            await send_long_message(channel, summary)
            break
          except json.JSONDecodeError:
            logging.error("Failed to decode JSON from result response")
            logging.error(f"Result response text: {result_response.text}")
            break
        else:
          logging.error("Empty response received from result URL")
          break


async def send_long_message(channel, message):
  max_length = 2000
  chunks = [
    message[i:i + max_length] for i in range(0, len(message), max_length)
  ]
  for chunk in chunks:
    await channel.send(chunk)


def retrieve_answer(vectorstore):
 #llm = ChatOpenAI(model_name="gpt-4")
  llm = ChatOpenAI()    
  #if message.content.startswith('!hf'): 
  #  llm = HuggingFaceHub(repo_id="tiiuae/falcon-40b", task="summarization", model_kwargs={"temperature":0.5, "max_length":1512})
  #  #llm = HuggingFacePipeline.from_model_id(model_id="tiiuae/falcon-40b-instruct", task="summarization", model_kwargs={"temperature":0, "max_length":64})
  #  logging.info("SETTING MODEL TO HUGGING FACE")
    
  if os.getenv('DEV_MODE'):
    wandb_config = {"project": "wandb_prompts_quickstart"}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(),callbacks=[WandbTracer(wandb_config)])
  else:
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
  query = "You are a helpful assistant with concise and accurate responses given in the tone of a professional presentation. Give and extremely detailed Summary of this document, including THE title  AND THE AUTHORS and ALL the IMPORTANT IDEAS expressed in the document as bullet points in markdown format"
  answer=qa.run(query)
  logging.info(answer)
  return answer

def create_directories_if_not_exists(pdf_path):
  if not os.path.exists(pdf_path):
    # If it doesn't exist, create it
    os.makedirs(pdf_path)
    print(f"Directory '{pdf_path}' created")
  else:
      print(f"Directory '{pdf_path}' already exists") 
   
   
def extract_yt_transcript(url):
    """
    Function to extract the YouTube video ID from a URL.
    """
    parsed_url = urlparse.urlparse(url)
    video_id = urlparse.parse_qs(parsed_url.query)['v'][0]

    transcriptlist = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = json.dumps(transcriptlist, indent=4)

    api_key = os.getenv('GOOGLE_API_KEY')
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.videos().list(
        part="snippet",
        id=video_id
    )
    response = request.execute()
    creator = response['items'][0]['snippet']['channelTitle']
    transcript_str = 'The creator of this video is '+ creator + '\n\n' + transcript
    
    return transcript_str

def callOPenAI():
  if os.getenv('DEV_MODE'):
   return RetrievalQA.from_chain_type(llm=ChatOpenAI(), chain_type="stuff", retriever=vectorstore.as_retriever(),callbacks=[WandbTracer(wandb_config)])
  else:
    return RetrievalQA.from_chain_type(llm=ChatOpenAI(), chain_type="stuff", retriever=vectorstore.as_retriever())


@bot.event
async def on_ready():
  logging.info(f'{bot.user} has connected to Discord!')

#Event handler for bot messages. Could break into smaller functions.
@bot.event
async def on_message(message):
  if message.author == bot.user:
    return
  global vectorstore
  if os.getenv('DEV_MODE'):
    global wandb_config
    wandb_config = {"project": "wandb_prompts_quickstart"}
  logging.info(f"MESSAGE RECEIVED!")
  
  if message.author == bot.user:
    return
  #Process PDFs 

  pdf_path = './docs/pdfs'
  create_directories_if_not_exists(pdf_path)
  web_doc_path = './docs/web/download.html'
  create_directories_if_not_exists('./docs/web')




  if 'http://' in  message.content or 'https://' in message.content:
    chunks = None
    if 'youtube.com' in message.content or 'youtu.be' in message.content:
        raw_text = extract_yt_transcript(message.content)
        chunks = get_text_chunks(''.join(raw_text))
        vectorstore = get_vectorstore(chunks)
        answer = retrieve_answer(vectorstore=vectorstore)
        await send_long_message(message.channel, answer)
    else:        
        urls = extract_url(message.content)   
        for url in urls:
            download_html(url, web_doc_path)
            loader = BSHTMLLoader(web_doc_path)
            data = loader.load()
        
            for page_info in data:
                chunks = get_text_chunks(page_info.page_content)
                vectorstore = get_vectorstore(chunks)
                answer = retrieve_answer(vectorstore=vectorstore)
                os.remove(web_doc_path) # Change here
                await send_long_message(message.channel, answer)


  if message.attachments:
    vectorstore=None
    for attachment in message.attachments:
        if attachment.filename.endswith('.pdf'):  # if the attachment is a pdf
          data = await attachment.read()  # read the content of the file
          with open(os.path.join(pdf_path, attachment.filename), 'wb') as f:  # save the pdf to a file
              f.write(data)
          raw_text = get_pdf_text(pdf_path)
          chunks = get_text_chunks(raw_text)
          vectorstore = get_vectorstore(chunks)
          answer = retrieve_answer(vectorstore=vectorstore)
        await send_long_message(message.channel, answer)
        return
  else:
    user_prompt = message.content
    logging.info(f"Received message from {message.author.name}: {user_prompt}")

    if vectorstore is None:
      #is DISCORD BOT MENTIONED in message?
      if bot.user in message.mentions:
        logging.info(f'{bot.user} sent a message with no vector to dicord bot')
        user_prompt = message.content
        logging.info(f"Received message from {message.author.name}: {user_prompt}")

        system_prompt = f"You are a helpful assistant with concise and accurate responses. The current time is {get_current_date()}, and the person messaging you is {message.author.name}."
        # Initialize dialog context
        if message.author.id not in dialog_contexts:
          dialog_contexts[message.author.id] = DialogContext()
          dialog_contexts[message.author.id].add_message("system", system_prompt)

        # Parse temperature
        temp_match = re.search(r'::temperature=([0-9]*\.?[0-9]+)::', user_prompt)
        temperature = 1.0  # default temperature
        if temp_match:
          temperature = float(temp_match.group(1))
          user_prompt = re.sub(
            r'::temperature=([0-9]*\.?[0-9]+)::', '',
            user_prompt)  # remove temperature command from user_prompt

        # Parse model
        model_match = re.search(r'::model=(\w+[-]*\w+\.?\w*)::', user_prompt)
        model = "gpt-4"  # default model
        if model_match:
          model = model_match.group(1)
          user_prompt = re.sub(
            r'::model=(\w+[-]*\w+\.?\w*)::', '',
            user_prompt)  # remove model command from user_prompt

    

   
        # Add user message to dialog context
        ai_message, cost, elapsed_time = openAIGPTCall(
          dialog_contexts[message.author.id].get_messages(),
          model=model,
          temperature=temperature)
        logging.info(f"Generated AI message: {ai_message}")
        logging.info(f"AI message cost: {cost}, elapsed time: {elapsed_time}")
        dialog_contexts[message.author.id].add_message("assistant", ai_message)
#
        #await send_long_message(message.channel, ai_message)
    else:
        if os.getenv('DEV_MODE'):
          wandb_config = {"project": "wandb_prompts_quickstart"}
          qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), chain_type="stuff", retriever=vectorstore.as_retriever(),callbacks=[WandbTracer(wandb_config)])
        else:
          qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), chain_type="stuff", retriever=vectorstore.as_retriever())
        query = "You are a helpful assistant with concise and accurate responses given in the tone of a professional presentation. Give and extremely detailed Summary of this document, including a title and ALL the important ideas expressed in the document as bullet points in markdown format"
        answer=qa.run(user_prompt)
        logging.info(answer)

        await send_long_message(message.channel, answer)




bot.run(os.getenv('DISCORD_TOKEN'))
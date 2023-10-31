from discord.ext import commands, tasks
import asyncio, os, json, time, logging, wandb, discord, openai
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA, LLMSummarizationCheckerChain
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import HuggingFaceHub,OpenAI
from langchain import HuggingFacePipeline
from wandb.integration.langchain import WandbTracer
from langchain.document_loaders import BSHTMLLoader, WebBaseLoader
from summary_prompts import get_guidelines
from utils import extract_urls, download_html, get_text_from_pdf, get_documents_from_pdf, get_text_chunks, create_directories_if_not_exists, extract_yt_transcript, extract_text_from_htmls, unzip_website, download_pdf_paper_from_url, convert_pdf_to_images, images_2_OCR, split_text
from vector import get_vectorstore, get_history_vectorstore, persist_new_chunks
from qdrant_vector import get_Qvector_store, return_qdrant, get_Qvector_store_from_docs
from concurrent.futures import ThreadPoolExecutor
from wandb.sdk.data_types.trace_tree import Trace
import threading
from langchain.llms import Bedrock
import boto3
from qdrant_client.http.models import SearchRequest, Filter, Distance
from llm_gateway import return_llm, retrieve_answer, summarise_documents, generate_openai_response, summarise_with_claude
from sentence_transformers import SentenceTransformer



# Set up logging  
logging.basicConfig(level=logging.INFO)
#GET API KEYS FROM .ENV FILE
#claude.api_key = os.getenv('ANTHROPIC_API_KEY')
openai.api_key = os.getenv('OPENAI_API_KEY')
scraperapi_key = os.getenv('SCRAPER_API_KEY')
brave_search_api_key = os.getenv('BRAVE_SEARCH_API_KEY')
intents = discord.Intents.all()
bot = commands.Bot("/", intents=intents)
client = commands

# Global Variables
vectorstore=None
dialog_contexts = {}
vector_flag = True
wandb.init(project="DiscordChatBot")

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


def get_current_date():
  return str(time.strftime("%Y-%m-%d, %H:%M:%S"))




async def send_long_message(channel, message):
  max_length = 2000
  chunks = [
    message[i:i + max_length] for i in range(0, len(message), max_length)
  ]
  for chunk in chunks:
    await channel.send(chunk)




def retrieve_answer(vectorstore):
  llm = return_llm()  
  
  #if message.content.startswith('!hf'): 
  #  llm = HuggingFaceHub(repo_id="tiiuae/falcon-40b", task="summarization", model_kwargs={"temperature":0.5, "max_length":1512})
  #  #llm = HuggingFacePipeline.from_model_id(model_id="tiiuae/falcon-40b-instruct", task="summarization", model_kwargs={"temperature":0, "max_length":64})
  #  logging.info("SETTING MODEL TO HUGGING FACE")
    
  if os.getenv('DEV_MODE'):
    wandb_config = {"project": "DiscordChatBot"}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":20}),callbacks=[WandbTracer(wandb_config)])
   #LLMSummarizationCheckerChain.from_llm(llm, verbose=True, max_checks=2)
  else:
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":20}))
  retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":20})
  
  guidelines = get_guidelines()["cod_summary_bullets"]
  query = guidelines
  #query = "You are a helpful assistant with concise and accurate responses given in the tone of a professional presentation. Give and detailed Summary of this document making sure to include the following sections Title: Authors: Ideas: Conclusions:"
  documents = retriever.get_relevant_documents(query)

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

async def get_message_text_history(channel):
    messages = []
    async for msg in channel.history(limit=5):
        messages.append(f"{msg.author.name}: {msg.content}")
    return messages

@bot.event
async def on_ready():
  logging.info(f'{bot.user} has connected to Discord!')

def create_faiss_vector_from_data(chunks):
  #chunks = get_text_chunks(data)
  vectorstore = get_vectorstore(chunks)
  executor = ThreadPoolExecutor()
  future = executor.submit(persist_new_chunks, chunks)
  return vectorstore






@bot.event
async def on_raw_reaction_add(reaction):
    channel = bot.get_channel(reaction.channel_id)
    # skip DM messages
    if isinstance(channel, discord.DMChannel):
        return

    message = await channel.fetch_message(reaction.message_id)
    emoji = reaction.emoji
    guild = bot.get_guild(reaction.guild_id)
    user = guild.get_member(reaction.user_id)
    emoji_name = reaction.emoji.name
    reaction = discord.utils.get(message.reactions, emoji=emoji)
    await send_long_message(channel, f'{user.display_name} reacted with {emoji} with the name {emoji_name}')



#Event handl  er for bot messages. Could break into smaller functions.
@bot.event
async def on_message(message):
  load_dotenv()
  folders=json.loads(os.getenv('FOLDERS'))
  vectorpath = folders['VECTOR_FOLDER']
  pdf_path = folders['PDF_FOLDER']
  web_doc_path = folders['WEB_FOLDER']
  zip_path = folders['ZIP_FOLDER']
  vector_flag = True
  if message.content.startswith('faiss'): 
    vector_flag = False

  if message.author == bot.user:
    return
  
  if os.getenv('DEV_MODE'):
    global wandb_config
    wandb_config = {"project": "DiscordChatBot"}
  logging.info(f"MESSAGE RECEIVED!")
  
  if message.author == bot.user:
    return
  #Process PDFs 

  FOLDERS = json.loads(os.getenv('FOLDERS'))
  for folder_name, folder_path in FOLDERS.items():
    create_directories_if_not_exists(folder_path)

  #create_directories_if_not_exists(pdf_path)
  #web_doc = web_doc_path +'/download.html'
  #create_directories_if_not_exists('docs/web')
  #create_directories_if_not_exists(zip_path)
  #create_directories_if_not_exists(vectorpath)





# IN CASE OF WEBSITES
  if 'http://' in  message.content or 'https://' in message.content:
    chunks = None
    if 'youtube.com' in message.content or 'youtu.be' in message.content:
        raw_text = extract_yt_transcript(message.content)
        response = summarise_with_claude(texts=raw_text)
        await send_long_message(message.channel, response)
        return
    elif 'arxiv.org' in message.content:
        raw_text = message.content
        chunks, docs, pdf_path, paper_number = download_pdf_paper_from_url(message.content)
        response = summarise_with_claude(texts=chunks)
        #response = summarise_documents(docs=docs)
        #answer = await response
        #img_path = convert_pdf_to_images(pdf_path)
        #chunks = images_2_OCR(img_path, paper_number)
        #vectorstore = get_Qvector_store(chunks) if vector_flag else create_faiss_vector_from_data(chunks)

        #answer = retrieve_answer(vectorstore)
        #remove_pdf = os.remove(img_path)
        
        await send_long_message(message.channel, response)
        return
    else:        
        urls = extract_urls(message.content)
        for url in urls:
            loader = WebBaseLoader(url)
            docs = loader.load()
            vectorstore = get_Qvector_store_from_docs(docs)
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
            chain = load_summarize_chain(llm, chain_type="stuff")
            answer = chain.run(docs)
            #loader = BSHTMLLoader#web_doc)
            #data = loader.load()

            #for pag#_info in data#
            #    page_text=page_in#o.page_content
            #    
            #   #vectorstore = get#Qvector_store(split_text(page_info.page_content)) if vector_flag else create_faiss_vector_from_data(split_text(page_info.page_content))
            #answer =             #er(vectorstore=vectorstore)
            #os.remove(            hange here
            await send_long_message(message.channel, answer)

        return       
  if message.attachments:
    for attachment in message.attachments:
        vectorstore = None
        if attachment.filename.endswith('.pdf'):  # if the attachment is a pdf
          data = await attachment.read()  # read the content of the file
          with open(os.path.join(pdf_path, attachment.filename), 'wb') as file:  # save the pdf to a file
              file.write(data)

          raw_text = get_text_from_pdf(pdf_path)
          answer = summarise_with_claude(raw_text=raw_text)
          #raw_text = get_text_from_pdf(pdf_path)
          ##docs = get_documents_from_pdf(pdf_path)
          #if vector_flag:
          #  vectorstore = get_Qvector_store(raw_text) 
          #else: 
          #   vectorstore = create_faiss_vector_from_data(raw_text)
          #answer = retrieve_answer(vectorstore)
          await send_long_message(message.channel, answer)
          return
        elif attachment.filename.endswith('.zip'):  # if the attachment is a pdf
          data = await attachment.read()  # read the content of the file
          with open(os.path.join(zip_path, attachment.filename), 'wb') as f:  # save the pdf to a file
              f.write(data)
          unzip_website(zip_path+"/"+attachment.filename, zip_path)
          executor = ThreadPoolExecutor()
          bg_thread = threading.Thread(target=extract_text_from_htmls, args=(zip_path,))
          bg_thread.start()
          #extract_text_from_htmls(zip_path)
          #future = executor.submit(extract_text_from_htmls, zip_path)
          return

          
  else:
    original_question = message.content
    encoder = SentenceTransformer('all-MiniLM-L6-v2') 
    logging.info(f"Received message from {message.author.name}: {original_question}")
    #context = await get_message_text_history(message.channel)
    context = []
    query = get_guidelines()["std_qa_prompt"]

    
    vectorstore = return_qdrant()
    hits = vectorstore.search(
    search_type="similarity",
    query=original_question,
    k=5
    )

    hits_string = "\n".join([f"Vector ID: {vector_id}, Similarity Score: {similarity_score}" for vector_id, similarity_score in hits])

    # Run the similarity search
    

    # `results` is a list of tuples, where each tuple is (vector_id, similarity_score)
    for vector_id, similarity_score in hits:
        print(f"Vector ID: {vector_id}, Similarity Score: {similarity_score}")
    
    hits_string = "\n".join([f"Vector ID: {vector_id}, Similarity Score: {similarity_score}" for vector_id, similarity_score in hits])

    context.append(hits_string)
    query = query.format(context="\n".join(context), question=original_question)


    answer = generate_openai_response(query)

    #if os.getenv('DEV_MODE'):
    #  wandb_config = {"project": "DiscordChatBot"}
    #  qa = RetrievalQA.from_chain_type(llm=return_llm(), chain_type="stuff", retriever=vectorstore.as_retriever(),callbacks=[WandbTracer(wandb_config)],return_source_documents = True)
    #else:
    #  qa = RetrievalQA.from_chain_type(llm=return_llm(), chain_type="stuff", retriever=vectorstore.as_retriever(),return_source_documents = True)
#
    ##result=qa({"query":query})
    #result=qa({"question":query})
    #answer = result['result']
    #sources = result['source_documents']
#
    #for source in sources:
    #   print(source.page_content)
   #
    #logging.info(answer)

    

    await send_long_message(message.channel, answer)
    return

    #await send_long_message(message.channel, answer)





bot.run(os.getenv('DISCORD_TOKEN'))
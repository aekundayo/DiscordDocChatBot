from discord.ext import commands, tasks
import asyncio, os, json, time, logging, discord, openai
import openai 
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA, LLMSummarizationCheckerChain
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import HuggingFaceHub,OpenAI
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.document_loaders import BSHTMLLoader, WebBaseLoader
from summary_prompts import get_guidelines
from utils import extract_urls, download_html, get_text_from_pdf, get_documents_from_pdf, get_text_chunks, create_directories_if_not_exists, extract_yt_transcript, extract_text_from_htmls, unzip_website, download_pdf_paper_from_url, split_text, convert_site_to_pdf, list_resource_groups, list_resources_and_total_cost, get_subscription_id, calculate_aws_bill, get_current_date
from vector import get_vectorstore, get_history_vectorstore, persist_new_chunks
from qdrant_vector import get_Qvector_store, return_qdrant, get_Qvector_store_from_docs
from concurrent.futures import ThreadPoolExecutor
import threading
from langchain.llms import Bedrock
import boto3
from qdrant_client.http.models import SearchRequest, Filter, Distance
from llm_gateway import return_llm, retrieve_answer, summarise_documents, summarise_with_claude, summarise_with_gpt_turbo, create_gpt_response
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
assistant = None

class DialogContext:
#load dot env file
  load_dotenv()

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

#
#def update_assistant():
# 
#
#  try:
#      # Assuming the OpenAI API has an endpoint to create assistants which is not currently known
#      new_assistant = openai.Client().beta.assistants.create
#      print(f"New assistant 'biodun-clone' created with ID: {new_assistant['id']}")
#  except openai.error.OpenAIError as e:
#      print(e)  # Prints the API error message
#  except Exception as e:
#      print(e)  # Prints any other error message
#
#Event handl  er for bot messages. Could break into smaller functions.
@bot.event
async def on_message(message):
  assistant = None
  load_dotenv()
  
  vectorpath = os.getenv('VECTOR_FOLDER')
  pdf_path = os.getenv('PDF_FOLDER')
  web_doc_path = os.getenv('WEB_FOLDER')
  zip_path = os.getenv('ZIP_FOLDER')
  log_path = os.getenv('LOG_FOLDER')
  folders = [vectorpath, pdf_path, web_doc_path, zip_path]

  vector_flag = True
  if message.content.startswith('faiss'): 
    vector_flag = False

  if message.author == bot.user:
    return
  

  if message.author == bot.user:
    return
  #Process PDFs 

  for folder_path in folders:
    create_directories_if_not_exists(folder_path)




# IN CASE OF WEBSITES
  if 'http://' in  message.content or 'https://' in message.content:
    chunks = None
    if 'youtube.com' in message.content or 'youtu.be' in message.content:
        raw_text = extract_yt_transcript(message.content)
        chunks = get_text_chunks(raw_text)
        get_Qvector_store(chunks)
        response = summarise_with_gpt_turbo(raw_text=raw_text)
        await send_long_message(message.channel, response)
        return
    elif 'arxiv.org' in message.content:
        raw_text = message.content
        chunks, docs, pdf_path, paper_number = download_pdf_paper_from_url(message.content)
        get_Qvector_store(chunks)
        response = summarise_with_gpt_turbo(texts=chunks)

        
        await send_long_message(message.channel, response)
        return
    else:        
        urls = extract_urls(message.content)
        for url in urls:
            loader = WebBaseLoader(url)
            docs = loader.load()
            if len(docs) <=1: 
              import requests

              res = requests.get(url)
              html_page = res.content

              from bs4 import BeautifulSoup

              soup = BeautifulSoup(html_page, 'html.parser')
              texts = soup.find_all(text=True)
              vectorstore = get_Qvector_store(texts)
              answer = summarise_with_gpt_turbo(texts=texts)
            else:
              vectorstore = get_Qvector_store_from_docs(docs)
              answer = summarise_with_gpt_turbo(docs=docs)
            #llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
            #chain = load_summarize_chain(llm, chain_type="stuff")
            #answer = chain.run(docs)

            await send_long_message(message.channel, answer)

        return       
  if message.attachments:
    for attachment in message.attachments:
        vectorstore = None
        if attachment.filename.endswith('.ogg'):  # if the attachment is a pdf
          data = await attachment.read()  # read the content of the file
          with open(os.path.join(pdf_path, attachment.filename), 'wb') as file:  # save the pdf to a file
              file.write(data)
              return
        if attachment.filename.endswith('.pdf'):  # if the attachment is a pdf
          data = await attachment.read()  # read the content of the file
          with open(os.path.join(pdf_path, attachment.filename), 'wb') as file:  # save the pdf to a file
              file.write(data)

          raw_text = get_text_from_pdf(pdf_path)
          chunks = get_text_chunks(raw_text)
          #get_Qvector_store(chunks)
          answer = summarise_with_gpt_turbo(raw_text=raw_text)

          await send_long_message(message.channel, answer)
          return
        elif attachment.filename.endswith('.zip'):  # if the attachment is a pdf
          data = await attachment.read()  # read the content of the file
          with open(os.path.join(zip_path, attachment.filename), 'wb') as f:  # save the pdf to a file
              f.write(data)
          unzip_website(zip_path+"/"+attachment.filename, zip_path)
          convert_site_to_pdf(zip_path)
          #extract_text_from_htmls(zip_path)
          #future = executor.submit(extract_text_from_htmls, zip_path)
          return

          
  else:
    original_question = message.content
    #encoder = SentenceTransformer('all-MiniLM-L6-v2') 
    #logging.info(f"Received message from {message.author.name}: {original_question}")
    ##context = await get_message_text_history(message.channel)
    #context = []
    #query = get_guidelines()["std_qa_prompt"]
#
    #
    #vectorstore = return_qdrant()
    #hits = vectorstore.search(
    #search_type="similarity",
    #query=original_question,
    #k=5
    #)
#
    #hits_string = "\n".join([f"Vector ID: {vector_id}, Similarity Score: {similarity_score}" for vector_id, similarity_score in hits])
#
    ## Run the similarity search
    #
#
    ## `results` is a list of tuples, where each tuple is (vector_id, similarity_score)
    #for vector_id, similarity_score in hits:
    #    print(f"Vector ID: {vector_id}, Similarity Score: {similarity_score}")
    #
    #hits_string = "\n".join([f"Vector ID: {vector_id}, Similarity Score: {similarity_score}" for vector_id, similarity_score in hits])
#
    #context.append(hits_string)
    #query = query.format(context="\n".join(context), question=original_question)
    openai_client = openai.Client()
   
    assistant = OpenAIAssistant(openai_client, original_question)
    

    answer = await assistant.block_on_inprogress_runs(assistant.thread.id)
  






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


import openai
import asyncio
class OpenAIAssistant:
    def __init__(self, client, original_question, polling_interval=2, response=None):
        self.client = client
        self.polling_interval = polling_interval
        self.assistant = ""
        my_assistants = client.beta.assistants.list(
          order="desc",
          limit="100",
        )
        for stored_assistant in my_assistants:
          if stored_assistant.name == os.environ["DEVOPS_ASSISTANT_NAME"]:
              self.assistant = stored_assistant
              self.assistant = client.beta.assistants.update(
                assistant_id=self.assistant.id,
                instructions="You are a helpful assistant and a knowledgeable expert on software design development and delivery engineering practices",
                model="gpt-4-1106-preview",
                name = os.environ["DEVOPS_ASSISTANT_NAME"],
                tools=[{
                    "type": "function",
                    "function": {
                      "name": "list_resource_groups",
                      "description": "Get Resource Groups From Subscription ID",
                      "parameters": {
                        "type": "object",
                        "properties": {
                          "subscription_id": {"type": "string", "description": "Subscription ID of Azure Subscription Containing Resource Groups"}
                        },
                        "required": ["subscription_id"]
                      }
                    }
                  },{
                    "type": "function",
                    "function": {
                      "name": "get_subscription_id",
                      "description": "Get Subscription ID from Subscription Name",
                      "parameters": {
                        "type": "object",
                        "properties": {
                          "subscription_name": {"type": "string", "description": "Name of the Subscription"}
                        },
                        "required": ["subscription_name"]
                      }
                    }
                  }
                  , {
                    "type": "function",
                    "function": {
                      "name": "calculate_aws_bill",
                      "description": "Get the Cost of AWS Billing",
                      "parameters": {
                        "type": "object",
                        "properties": {
                          "start_date": {"type": "string", "description": "The start date of the period for measuring cost yyyy-mm-dd"},
                          "end_date": {"type": "string", "description": "The end date of the period for measuring cost yyyy-mm-dd"},
                        },
                        "required": []
                      }
                    }
                  }, {
                    "type": "function",
                    "function": {
                      "name": "list_resources_and_total_cost",
                      "description": "Get the cost of resources in Resource Group",
                      "parameters": {
                        "type": "object",
                        "properties": {
                          "resource_group_name": {"type": "string", "description": "The name of the resource group to be costed"},
                        },
                        "required": ["resource_group_name"]
                      }
                    }
                  }, {
                    "type": "function",
                    "function": {
                      "name": "get_current_date",
                      "description": "Get the Current Date of the System",
                      "parameters": {
                      }
                    }
                  }]
            )
        
        #if self.assistant=="":
        #   self.assistant = client.beta.assistants.create(
        #        instructions="You are a helpful assistant and a knowledgeable expert on software design development and delivery engineering practices",
        #        model="gpt-4-1106-preview",
        #        name = os.environ["DEVOPS_ASSISTANT_NAME"],
        #        tools=[{
        #            "type": "function",
        #            "function": {
        #              "name": "list_resource_groups",
        #              "description": "Get Resource Groups From Subscription ID",
        #              "parameters": {
        #                "type": "object",
        #                "properties": {
        #                  "subscription_id": {"type": "string", "description": "Subscription ID of Azure Subscription Containing Resource Groups"}
        #                },
        #                "required": ["subscription_id"]
        #              }
        #            }
        #          },{
        #            "type": "function",
        #            "function": {
        #              "name": "get_subscription_id",
        #              "description": "Get Subscription ID from Subscription Name",
        #              "parameters": {
        #                "type": "object",
        #                "properties": {
        #                  "subscription_name": {"type": "string", "description": "Name of the Subscription"}
        #                },
        #                "required": ["subscription_name"]
        #              }
        #            }
        #          }
        #          , {
        #            "type": "function",
        #            "function": {
        #              "name": "calculate_aws_bill",
        #              "description": "Get the Cost of AWS Billing",
        #              "parameters": {
        #                "type": "object",
        #                "properties": {
        #                  "start_date": {"type": "string", "description": "The start date of the period for measuring cost yyyy-mm-dd"},
        #                  "end_date": {"type": "string", "description": "The end date of the period for measuring cost yyyy-mm-dd"},
        #                },
        #                "required": []
        #              }
        #            }
        #          }, {
        #            "type": "function",
        #            "function": {
        #              "name": "list_resources_and_total_cost",
        #              "description": "Get the cost of resources in Resource Group",
        #              "parameters": {
        #                "type": "object",
        #                "properties": {
        #                  "resource_group_name": {"type": "string", "description": "The name of the resource group to be costed"},
        #                },
        #                "required": ["resource_group_name"]
        #              }
        #            }
        #          }, {
        #            "type": "function",
        #            "function": {
        #              "name": "get_current_date",
        #              "description": "Get the Current Date of the System",
        #              "parameters": {
        #              }
        #            }
        #          }]
        #    )
         

            
        
        #if not hasattr(self, 'thread'):
        self.thread = self.client.beta.threads.create(
          messages=[
            {
              "role": "user",
              "content": f"{original_question}"
            }
          ]
        )

        self.run = self.client.beta.threads.runs.create(
          thread_id=self.thread.id,
          assistant_id=self.assistant.id
        )

    async def submit_tool_outputs(self, thread_id, run_id, tool_outputs):
        tool_outputs_str = json.dumps(tool_outputs, indent=2)

        formatted_tool_outputs = [{"tool_call_id": k, "output": str(v)} for k, v in tool_outputs.items()]
        return self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run_id,
            tool_outputs=formatted_tool_outputs
              )


    async def wait_for_run(self, thread_id, run_id, handle_actions=False):
        # Mapping of function names to function objects
        function_handlers = {
            "get_subscription_id": get_subscription_id,
            "list_resources_and_total_cost": list_resources_and_total_cost,
            "calculate_aws_bill": calculate_aws_bill,
            "list_resource_groups": list_resource_groups,
            "get_current_date": get_current_date
        }

        while True:
            await asyncio.sleep(self.polling_interval)
            self.run = self.client.beta.threads.runs.retrieve(thread_id=self.thread.id, run_id=self.run.id)
            
            tool_outputs ={}
            # Dynamic function calling
            if self.run.status == 'requires_action' and self.run.required_action.type == "submit_tool_outputs":
                for tool_call in self.run.required_action.submit_tool_outputs.tool_calls:
                    if tool_call.type == "function":
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)

                        if function_name in function_handlers:
                            # Call the function with unpacked arguments
                            tool_outputs[tool_call.id] = function_handlers[function_name](**function_args)

                await self.submit_tool_outputs(thread_id, run_id, tool_outputs)
                            
     
            elif self.run.status in ['cancelled', 'failed', 'completed', 'expired']:
                message=None
               
                if  self.run.status == 'completed':
                   thread_messages = self.client.beta.threads.messages.list(self.thread.id)
                   self.response = thread_messages.data[0].content[0].text.value
                   #message_id = thread_messages.last_id
                   #message = self.client.beta.threads.messages.retrieve(message_id=message_id, thread_id=self.thread.id)
                   #if len(message.content) > 0:
#
                   # if message.content[0].type=='text':
                   #   response = message.content[0].content 
                return self.response

    async def retrieve_last_run(self, thread_id):
      self.run = self.client.beta.threads.runs.retrieve(
        thread_id=self.thread.id,
        run_id=self.run.id
      )

      return self.run

        # Implement this method based on your application logic
        

    def is_run_completed(self, run):
        return run.status in ['completed', 'cancelled', 'failed', 'expired']

    async def block_on_inprogress_runs(self, thread_id):
        
        while True:
            run = await self.retrieve_last_run(self.thread.id)
            if not run or self.is_run_completed(run):
                return self.response
            await self.wait_for_run(thread_id, run.id)

bot.run(os.getenv('DISCORD_TOKEN'))
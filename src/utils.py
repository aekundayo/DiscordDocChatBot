import os
import requests
import re
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
import urllib.parse as urlparse
from googleapiclient.discovery import build
import zipfile
from bs4 import BeautifulSoup
import threading
from vector import persist_new_chunks
from qdrant_vector import update_qdrant

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

def unzip_website(file_name, extract_folder="extracted_website"):
    # Unzipping the website
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

async def extract_text_from_htmls(folder):
    # Iterate through all files in the folder
    for subdir, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".html"):
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f, 'html.parser')
                    # Extract text from the HTML
                    text = soup.get_text()
                    # Print or save the text as needed
                    print(f"Text from {file_path}:\n{text}\n{'='*40}\n")
                    chunks = get_text_chunks(text)
                    #persist_new_chunks(chunks)
                    update_qdrant(chunks)
    
    remove_folder_contents(folder)
                    

def remove_folder_contents(path):
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        
        # If it's a file, remove it directly using os.remove
        if os.path.isfile(filepath):
            os.remove(filepath)
        # If it's a directory, call the function recursively
        elif os.path.isdir(filepath):
            remove_folder_contents(filepath)
            os.rmdir(filepath)

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
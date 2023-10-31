import os, cv2
import requests
import re
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, NLTKTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
import urllib.parse as urlparse
from googleapiclient.discovery import build
import zipfile
from bs4 import BeautifulSoup
import threading
from vector import persist_new_chunks
from qdrant_vector import update_qdrant
from langchain.document_loaders import PDFMinerLoader
from langchain.docstore.document import Document 
from pdf2image import convert_from_path, convert_from_bytes
import layoutparser as lp
import torch



def extract_urls(s):
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

def extract_text_from_htmls(folder):
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

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE")), # Specify the character chunk sizecz
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP")), # "Allowed" Overlap across chunks
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks              

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

def get_documents_from_pdf(path):
    # define the path
    documents  =   []
    for filename in os.listdir(path):
    # check if the file is a pdf
      if filename.endswith('.pdf'):
        text = ""
        with open(os.path.join(path, filename), 'rb') as pdf_doc:
        
            docs    =   PDFMinerLoader(f"{pdf_doc.name}").load()
            text_splitter   =   RecursiveCharacterTextSplitter(
            chunk_size=os.getenv("CHUNK_SIZE"), # Specify the character chunk sizecz
            chunk_overlap=os.getenv("CHUNK_OVERLAP"), # "Allowed" Overlap across chunks
            length_function=len # Function used to evaluate the chunk size (here in terms of characters)
            )     


            documents    =   text_splitter.split_documents(docs)
            os.remove(os.path.join(path, filename))
    return documents

def get_text_from_pdf(path):
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

        
def images_2_OCR(imgs_paths, paper_number):
    docs        =   []
    for img_path_idx in range(len(os.listdir(imgs_paths))):
        img_path        =   os.path.join(imgs_paths, "page{}.jpg".format(img_path_idx))
        page_content    =   extract_text_pdf_image_PubLay_OCR(img_path)
        for content in page_content:
            text    =   content[0]
            cat     =   content[1]
            if "REFERENCES" in text and cat == "Title":
                return docs
            metadata        =   {"page_number" : img_path_idx, "category" : cat, "source" : paper_number}
            docs.append(Document(page_content=text, metadata=metadata))

    return docs

def download_pdf_paper_from_url(url):
    paper_number    =   os.path.basename(url).strip(".pdf")
    pdf_url = f"https://arxiv.org/pdf/{paper_number}.pdf"
    res             =   requests.get(pdf_url)
    pdf_folder_path = json.loads(os.getenv('FOLDERS'))['PDF_FOLDER']
    pdf_path        =   f"{pdf_folder_path}/{paper_number}.pdf"
    with open(pdf_path, 'wb') as f:
        f.write(res.content)
    docs    =   PDFMinerLoader(f"{pdf_path}").load()
    text_splitter   =   RecursiveCharacterTextSplitter(
    chunk_size=int(os.getenv("CHUNK_SIZE")), # Specify the character chunk sizecz
    chunk_overlap=int(os.getenv("CHUNK_OVERLAP")), # "Allowed" Overlap across chunks
    length_function=len # Function used to evaluate the chunk size (here in terms of characters)
    )     

    docs    =   text_splitter.split_documents(docs)
    texts = [doc.page_content for doc in docs]
    os.remove(pdf_path)
    return texts, docs, pdf_path, paper_number



ocr_agent                   =   lp.TesseractAgent(languages="eng")

model_publay    =   lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.6],
                    label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})

def convert_pdf_to_images(pdf_path):
    img_folder    = f"{json.loads(os.getenv('FOLDERS'))['PDF_FOLDER']}/images"
    #img_path    =   os.path.join(os.getenv("ZIP_FOLDER"), os.path.basename(pdf_path).strip(".pdf") + "_images")
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    images      =   convert_from_path(pdf_path, output_folder=img_folder)
    for i in range(len(images)):
        images[i].save(os.path.join(img_folder, "page" + str(i) + ".jpg"), "JPEG")
    print("Images Saved !")
    
    model_publay    =   lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.6],
                    label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
    page_idx    =   6
    img_path    =   os.path.join(img_folder, f"page{page_idx}.jpg")
    img         =   cv2.imread(img_path)
    img         =   img[..., ::-1]
    layout  =   model_publay.detect(img)
    lp.draw_box(img, layout)
    return img_folder

def extract_text_pdf_image_PubLay_OCR(img_path):
    texts       =   []
    image       =   cv2.imread(img_path)
    if image is None:
        print("Could not read the image.")
        return texts
    image       =   image[..., ::-1]
    layout      =   model_publay.detect(image)
    text_blocks =   lp.Layout([b for b in layout if b.type in ['Text', 'List', 'Title']])

    # Organize text blocks based on their positions on the page
    h, w            =   image.shape[:2]
    left_interval   =   lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(image)
    left_blocks     =   text_blocks.filter_by(left_interval, center=True)
    left_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)

    right_blocks            =   lp.Layout([b for b in text_blocks if b not in left_blocks])
    right_blocks.sort(key   =   lambda b:b.coordinates[1], inplace=True)

    text_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])

    for layout_i in text_blocks:    # If some of the blocks overlap -> Take the one with the most associated area
        for layout_j in text_blocks:
            if layout_i != layout_j:
                refine_bboxes(layout_i, layout_j)

    for block in text_blocks:
        segment_image = (block
                        .pad(left=5, right=5, top=5, bottom=5)
                        .crop_image(image))
            # add padding in each image segment can help
            # improve robustness 
            
        text = ocr_agent.detect(segment_image)
        block.set(text=text, inplace=True)
    for l in text_blocks:
        texts.append([l.text, l.type])
    return texts

def get_coordinate(data):

  x1 = data.block.x_1
  y1 = data.block.y_1
  x2 = data.block.x_2
  y2 = data.block.y_2

  return torch.tensor([[x1, y1, x2, y2]], dtype=torch.float)

import cv2
import torch

def get_iou(box_1, box_2):
    # Convert torch tensors to lists
    box_1 = box_1[0].tolist()
    box_2 = box_2[0].tolist()

    # Convert to integer coordinates as OpenCV expects them as integers
    box_1 = [int(coord) for coord in box_1]
    box_2 = [int(coord) for coord in box_2]

    # Create rectangles from bounding boxes
    rect_1 = (box_1[0], box_1[1], box_1[2] - box_1[0], box_1[3] - box_1[1])
    rect_2 = (box_2[0], box_2[1], box_2[2] - box_2[0], box_2[3] - box_2[1])

    # Calculate intersection area
    intersection_area = (max(0, min(rect_1[0] + rect_1[2], rect_2[0] + rect_2[2]) - max(rect_1[0], rect_2[0])) *
                        max(0, min(rect_1[1] + rect_1[3], rect_2[1] + rect_2[3]) - max(rect_1[1], rect_2[1])))

    # Calculate union area
    union_area = rect_1[2] * rect_1[3] + rect_2[2] * rect_2[3] - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area != 0 else 0.0

    return torch.tensor([[iou]], dtype=torch.float)

# Example usage:
# Assuming box_1 and box_2 are torch tensors representing bounding boxes
# iou = get_iou(box_1, box_2)


def get_area(bbox):
  w = bbox[0, 2] - bbox[0, 0] # Width
  h = bbox[0, 3] - bbox[0, 1] # Height
  area  = w * h

  return area

def refine_bboxes(block_1, block_2):

  bb1 = get_coordinate(block_1)
  bb2 = get_coordinate(block_2)

  iou = get_iou(bb1, bb2)

  if iou.tolist()[0][0] != 0.0:

    a1 = get_area(bb1)
    a2 = get_area(bb2)

    block_2.set(type='None', inplace= True) if a1 > a2 else block_1.set(type='None', inplace= True)

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=0,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_sentences_chunks(text):
    text_splitter = NLTKTextSplitter()
    sentences = text_splitter.split_text(text)
    return sentences


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
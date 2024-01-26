import os
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

from qdrant_vector import update_qdrant
from langchain.document_loaders import PDFMinerLoader
from langchain.docstore.document import Document 
from pdf2image import convert_from_path, convert_from_bytes

import torch

from azure.mgmt.resource import ResourceManagementClient
from azure.identity import DefaultAzureCredential

import boto3

def get_daily_costs(start_date, end_date):
    try:
        # Call the Cost Explorer API to get the cost data
        client = boto3.client('ce', region_name='us-east-1')
        response = client.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity='DAILY',
            Metrics=['UnblendedCost']
        )
        
        # Process the response and print costs per day
        for result in response.get('ResultsByTime', []):
            print(f"Date: {result.get('TimePeriod').get('Start')} - Cost: {result.get('Total').get('UnblendedCost').get('Amount')} {result.get('Total').get('UnblendedCost').get('Unit')}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

def get_current_date():
    from datetime import datetime
    return datetime.now().strftime('%Y-%m-%d')

def calculate_aws_bill( start_date = None, end_date = None ):
    from datetime import datetime
    if not start_date:
        start_date = datetime.now().replace(day=1).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    client = boto3.client('ce', region_name='us-east-1')
    response = client.get_cost_and_usage(
        TimePeriod={
            'Start': start_date, #'2022-01-01'
            'End': end_date
        },
        Granularity='DAILY',
        Metrics=[
            'UnblendedCost',
        ],
        GroupBy=[
            {
                'Type': 'DIMENSION',
                'Key': 'SERVICE'
            },
        ]
    )
    return response


def get_subscription_id(subscription_name):
    subscription_mappings=json.loads(os.getenv('SUBSCRIPTION_MAPPINGS'))
    return subscription_mappings[subscription_name]


def list_resource_groups(subscription_id):
    # Get credentials
    credential = DefaultAzureCredential()
    # Instantiate a resource management client
    resource_client = ResourceManagementClient(credential, subscription_id)

    # List all resource groups in your subscription
    resource_groups = resource_client.resource_groups.list()
    resource_list = []
    # Print all resource groups
    for resource_group in resource_groups:
        #print(resource_group.name)
        
        resource_list.append(resource_group.name)
    
    resources_dict = {"success": "true", "resource_groups": resource_list}
    resources_json = json.dumps(resources_dict)
    print(resources_json)
    return resources_json

from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.costmanagement import CostManagementClient

def list_resources_and_total_cost(resource_group_name):
    # Get credentials
    credential = DefaultAzureCredential()

    subscription_id = "22e53644-ba7c-4d73-a775-e9724eea3d86"
    # Instantiate a resource management client
    resource_client = ResourceManagementClient(credential, subscription_id)

    # Instantiate a cost management client
    cost_client = CostManagementClient(credential)

    # List all resources in the specified resource group
    resources = resource_client.resources.list_by_resource_group(resource_group_name)
    resource_list = []
    # For each resource, get the cost
    for resource in resources:
        resource_list.append(f"resource: {resource.name} kind: {resource.kind} type: {resource.type} created: {resource.created_time}"  ) 

    # Define the scope (in this case, a resource group)
    scope = f"/subscriptions/{subscription_id}/resourceGroups/{resource_group_name}"

    # Define the parameters for the cost query (timeframe, type, dataset, etc.)
    cost_query_parameters = {
        "type": "Usage",
        "timeframe": "MonthToDate",
        "dataset": {
            "aggregation": {
                "totalCost": {
                    "name": "PreTaxCost",
                    "function": "Sum"
                }
            },
            "grouping": [
                {
                    "type": "Dimension",
                    "name": "ResourceGroupName"
                }
            ]
        }
    }

    # Execute the cost query
    cost_query_results = cost_client.query.usage(scope, cost_query_parameters)
        
    # The cost query result is better handled as follows:
    #print(cost_query_results)

    # Access the rows property of the cost_query_results object
    
    rows = cost_query_results.rows

    # Test if rows has values
    if rows:
        # Now you can index into rows as if it were a list
        cost_value = rows[0][0]
        currency_value = rows[0][2]
    else:
        cost_value = "0"
        currency_value = 'N/A'

    #print(f"Cost: {cost_value}, Currency: {currency_value}")
    
    cost = f"cost: {cost_value}"
    currency = f"currency: {currency_value}"

    resource_json = json.dumps(resource_list)
    resources_dict = {"success": "true", "resources": resource_list, "cost":cost_value, "currency": currency_value}
    print(resources_dict)
    return resources_dict


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
    pdf_folder_path = os.getenv('PDF_FOLDER')
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



#ocr_agent                   =   lp.TesseractAgent(languages="eng")
#
#model_publay    =   lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
#                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.6],
#                    label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
#
#def convert_pdf_to_images(pdf_path):
#    img_folder    = f"{json.loads(os.getenv('FOLDERS'))['PDF_FOLDER']}/images"
#    #img_path    =   os.path.join(os.getenv("ZIP_FOLDER"), os.path.basename(pdf_path).strip(".pdf") + "_images")
#    if not os.path.exists(img_folder):
#        os.makedirs(img_folder)
#    images      =   convert_from_path(pdf_path, output_folder=img_folder)
#    for i in range(len(images)):
#        images[i].save(os.path.join(img_folder, "page" + str(i) + ".jpg"), "JPEG")
#    print("Images Saved !")
#    
#    model_publay    =   lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
#                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.6],
#                    label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
#    page_idx    =   6
#    img_path    =   os.path.join(img_folder, f"page{page_idx}.jpg")
#    img         =   cv2.imread(img_path)
#    img         =   img[..., ::-1]
#    layout  =   model_publay.detect(img)
#    lp.draw_box(img, layout)
#    return img_folder
#
#def extract_text_pdf_image_PubLay_OCR(img_path):
#    texts       =   []
#    image       =   cv2.imread(img_path)
#    if image is None:
#        print("Could not read the image.")
#        return texts
#    image       =   image[..., ::-1]
#    layout      =   model_publay.detect(image)
#    text_blocks =   lp.Layout([b for b in layout if b.type in ['Text', 'List', 'Title']])
#
#    # Organize text blocks based on their positions on the page
#    h, w            =   image.shape[:2]
#    left_interval   =   lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(image)
#    left_blocks     =   text_blocks.filter_by(left_interval, center=True)
#    left_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)
#
#    right_blocks            =   lp.Layout([b for b in text_blocks if b not in left_blocks])
#    right_blocks.sort(key   =   lambda b:b.coordinates[1], inplace=True)
#
#    text_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])
#
#    for layout_i in text_blocks:    # If some of the blocks overlap -> Take the one with the most associated area
#        for layout_j in text_blocks:
#            if layout_i != layout_j:
#                refine_bboxes(layout_i, layout_j)
#
#    for block in text_blocks:
#        segment_image = (block
#                        .pad(left=5, right=5, top=5, bottom=5)
#                        .crop_image(image))
#            # add padding in each image segment can help
#            # improve robustness 
#            
#        text = ocr_agent.detect(segment_image)
#        block.set(text=text, inplace=True)
#    for l in text_blocks:
#        texts.append([l.text, l.type])
#    return texts
#
#def get_coordinate(data):
#
#  x1 = data.block.x_1
#  y1 = data.block.y_1
#  x2 = data.block.x_2
#  y2 = data.block.y_2
#
#  return torch.tensor([[x1, y1, x2, y2]], dtype=torch.float)
#
#
#import torch
#
#def get_iou(box_1, box_2):
#    # Convert torch tensors to lists
#    box_1 = box_1[0].tolist()
#    box_2 = box_2[0].tolist()
#
#    # Convert to integer coordinates as OpenCV expects them as integers
#    box_1 = [int(coord) for coord in box_1]
#    box_2 = [int(coord) for coord in box_2]
#
#    # Create rectangles from bounding boxes
#    rect_1 = (box_1[0], box_1[1], box_1[2] - box_1[0], box_1[3] - box_1[1])
#    rect_2 = (box_2[0], box_2[1], box_2[2] - box_2[0], box_2[3] - box_2[1])
#
#    # Calculate intersection area
#    intersection_area = (max(0, min(rect_1[0] + rect_1[2], rect_2[0] + rect_2[2]) - max(rect_1[0], rect_2[0])) *
#                        max(0, min(rect_1[1] + rect_1[3], rect_2[1] + rect_2[3]) - max(rect_1[1], rect_2[1])))
#
#    # Calculate union area
#    union_area = rect_1[2] * rect_1[3] + rect_2[2] * rect_2[3] - intersection_area
#
#    # Calculate IoU
#    iou = intersection_area / union_area if union_area != 0 else 0.0
#
#    return torch.tensor([[iou]], dtype=torch.float)
#
## Example usage:
## Assuming box_1 and box_2 are torch tensors representing bounding boxes
## iou = get_iou(box_1, box_2)
#
#
#def get_area(bbox):
#  w = bbox[0, 2] - bbox[0, 0] # Width
#  h = bbox[0, 3] - bbox[0, 1] # Height
#  area  = w * h
#
#  return area
#
#def refine_bboxes(block_1, block_2):
#
#  bb1 = get_coordinate(block_1)
#  bb2 = get_coordinate(block_2)
#
#  iou = get_iou(bb1, bb2)
#
#  if iou.tolist()[0][0] != 0.0:
#
#    a1 = get_area(bb1)
#    a2 = get_area(bb2)
#
#    block_2.set(type='None', inplace= True) if a1 > a2 else block_1.set(type='None', inplace= True)

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
    try:
        # Create the directory
        os.makedirs(pdf_path)
        print(f"Directory {pdf_path} was created successfully.")
    except OSError as error:
        print(f"Creation of the directory {pdf_path} failed due to: {error}")
    else:
        print(f"The directory {pdf_path} already exists.")


def extract_yt_transcript(url):
    """
    Function to extract the YouTube video ID from a URL.
    """
    if "youtu.be" in url:
        video_id = url.split('/')[-1].split('?')[0]
    else:
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


import requests
import os

def download_and_convert_site(url = "https://docs.radapp.io/concepts/@https://docs.radapp.io/concepts/"):
    response = requests.get(url)
    response.raise_for_status()

    # Create the directory if it doesn't exist
    if not os.path.exists("downloaded_site"):
        os.makedirs("downloaded_site")

    # Save the initial page
    with open("downloaded_site/index.html", "w") as file:
        file.write(response.text)

    # Extract the base URL
    base_url = url[:url.rfind("/")+1]

    # Download subsequent pages
    links = extract_links(response.text)
    for link in links:
        page_url = base_url + link
        page_response = requests.get(page_url)
        page_response.raise_for_status()
    
        # Save the page
        page_path = "downloaded_site/" + link
        with open(page_path, "w") as file:
            file.write(page_response.text)

    print("Website downloaded and converted successfully")

def extract_links(text):
    import re

    links = re.findall(r'href=[\'"]?([^\'" >]+)', text)

    return links

import subprocess
import os

import os

def convert_site_to_pdf(parent_path):
    counter = 0
    for root, dirs, files in os.walk(parent_path):
        for file in files:
            if file.endswith(".html"):
                counter += 1
                html_filename = os.path.basename(file)
                pdf_filename = html_filename + str(counter) + ".pdf"
                html_path = os.path.join(root, file)
                pdf_path =  os.getenv('PDF_FOLDER')  + pdf_filename 
                convert_html_to_pdf(html_path, pdf_path)

def convert_html_to_pdf(input_filename, output_filename):
    try:
        # Construct the command to execute
        command = ["wkhtmltopdf", input_filename, output_filename]

        # Run the command
        subprocess.run(command, check=True)
        print(f"PDF generated successfully: {output_filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error during PDF generation: {e}")
    

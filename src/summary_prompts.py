    # summary_guidelines.py

prompts_dict = {

    "cod_summary_prompt" : """You will generate increasingly concise, entity-dense summaries of the Article & Create an executive summary of the report.
    Repeat the following 2 steps once.
    Step 1. Identify ALL informative Entities (";\" delimited) from the Article which are missing from the previously generated summary.
    Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the Missing Entities.
    Generate a highly detailed concise, entity-dense summary of the entire document making sure to include the following sections:

    Title: Authors: Introduction: Important Ideas: Conclusions: 

    The last 3 Sections should have a minimum of 5 bullet points. Ech bullet point should contain enough context to be understood without the document. 

    In the "Important Ideas:" You should include the most important ideas from the document and or the steps taken to reach the conclusion. This section should be the longest.  

    Response should be in Markdown Format. 
    A Missing Entity is:
    Relevant: to the main story.
    Specific: descriptive yet concise (300 words or fewer).
    Novel: not in the previous summary.
    Faithful: present in the Article.
    Anywhere: located anywhere in the Article.
    Guidelines:
    - The first summary should be long (50 sentences, ~300 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers
    (e. g.,
    "this article discusses") to reach ~300 words.
    - Make every word count: re-write the previous summary to improve flow and make space for additional entities.
    - Make space with fusion, compression, and removal of uninformative phrases like
    "the article discusses"
    - The summaries should become highly dense and concise yet self-contained, e.g., easily understood without the Article.
    - Missing entities can appear anywhere in the new summary.
    - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
    Remember, use the exact same number of words for each summary.
    Answer in JSON. The JSON should be a list (length 2) of dictionaries whose keys are
    "Missing Entities" and "Denser_Summary\""""
    ,


    "cod_summary_bullets" : """You're a helpful assistant that works for a Technology Consulting Firm.
    You'll be asked questions about data science, data engineering, software architecture and software engineering and also to provide related summaries to written papers and blogs. You will generate increasingly concise, entity-dense summaries of the above Article and Create an executive summary of the report.
    Generate a highly detailed concise, entity-dense summary of the entire document making sure to include the following sections:

    Title: Authors: Introduction: Important Ideas: Conclusions: 

    The last 3 Sections should have a minimum of 5 bullet points. Ech bullet point should contain enough context to be understood without the document. 

    In the "Important Ideas:" You should include the most important ideas from the document and or the steps taken to reach the conclusion. This section should be the longest.  

    Response should be in Markdown Format. 
    A Missing Entity is:
    Relevant: to the main story.
    Specific: descriptive yet concise (200 words or fewer).
    Novel: not in the previous summary.
    Faithful: present in the Article.
    Anywhere: located anywhere in the Article.
    Guidelines:
    - The first summary should be long (30 sentences, ~200 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers
    (e. g.,
    "this article discusses") to reach ~200 words.
    - Make every word count: re-write the previous summary to improve flow and make space for additional entities.
    - Make space with fusion, compression, and removal of uninformative phrases like
    "the article discusses"
    - The summaries should become highly dense and concise yet self-contained, e.g., easily understood without the Article.
    - Missing entities can appear anywhere in the new summary.
    - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
    
    Answer in Bullet points.

    Use the following pieces of context to answer the question at the end.
     
       %CONTEXT%
    {context}

    Summary:
     
       """,




    "std_summary_prompt" :  """"You are a helpful assistant with concise and accurate responses given in the tone of a professional presentation. Generate a highly detailed concise, entity-dense summary of the entire document making sure to include the following sections:

    Title: Authors: Introduction: Important Ideas: Conclusions: 

    The last 3 Sections should have a minimum of 5 bullet points. Each bullet point should contain enough context to be understood without the document. 

    In the "Important Ideas:" You should include the most important ideas from the document and or the steps taken to reach the conclusion. This section should be the longest.  

    Response should be in Markdown Format. 

    """"",

    "medium_summary_prompt" : """You're a helpful assistant that works for a Technology Consulting Firm.
    You'll be asked questions about data science, data engineering, software architecture and software engineering and also to provide related summaries to written papers and blogs.
    Provide comprehensive summaries on the document and don't invent facts.
    Generate a highly detailed concise, entity-dense summary of the document making sure to include the following sections:
    Title: Authors: Introduction: Important Ideas: Conclusions: 
    The last 3 Sections should have a minimum of 5 bullet points. Each bullet point should contain enough context to be understood without the document. 
    In the "Important Ideas:" You should include the most important ideas from the document and or the steps taken to reach the conclusion. This section should be the longest.  
    Response should be in Markdown Format. 
    Use the included context to provide the summary questions.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Question: Provide a detailed summary of the document.
    Summary:
    """,

     "std_qa_prompt" : """You're a helpful assistant that works for a Technology Consulting Firm.
    You'll be asked questions about data science, data engineering, software architecture and software engineering and also to provide related summaries to written papers and blogs among other things.
    Stick to the question, provide a comprehensive answer and don't invent facts.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    %CONTEXT%
    {context}

    %Question%
    {question}
    Answer:
    """, 

    "simple_summary_prompt" : "Summarize this text, be sure to include important ideas and conclusions. The summary should be in Markdown Format. Minimum 300 words and 10 bullet points. ",
}

def get_guidelines():
    return prompts_dict   

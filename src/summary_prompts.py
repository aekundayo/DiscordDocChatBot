# summary_guidelines.py

cod_summary_prompt = """You will generate increasingly concise, entity-dense summaries of the Article & Create an executive summary of the report.
Repeat the following 2 steps once.
Step 1. Identify ALL informative Entities (";\" delimited) from the Article which are missing from the previously generated summary.
Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the Missing Entities.
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


cod_summary_bullets = """You will generate increasingly concise, entity-dense summaries of the above Article and Create an executive summary of the report.
Repeat the following 2 steps once.
Step 1. Identify ALL informative Entities (";\" delimited) from the Article which are missing from the previously generated summary.
Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the Missing Entities.
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
Remember, use the exact same number of words for each summary.
Answer in Bullet points. There should be a list for both summarisation attempts with sections "Missing Entities" and "Denser_Summary\""""

std_summary_prompt =  """"You are a helpful assistant with concise and accurate responses given in the tone of a professional presentation. Generate increasingly concise, entity-dense summaries of this document making sure to include butnot be limited to the following sections:

Title: Authors: Introduction: Important Ideas: Conclusions: 

The last 3 Sections should have a minimum of 5 bullet points.

In the "Important Ideas:" You should include the most important ideas from the document and or the steps taken to reach the conclusion. This section should be the longest.  

Response should be in Markdown Format. 
"""""

std_qa_prompt = "You are a helpful assistant with concise and accurate responses given in the tone of a professional presentation. Try and answer the question as truthfully as possible. What is the answer to the question: "

def get_guidelines():
    return cod_summary_prompt, cod_summary_bullets, std_summary_prompt, std_qa_prompt   

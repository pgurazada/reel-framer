import json
import requests


with open('creds.json', 'r') as as_creds:
    data = as_creds.read()

creds = json.loads(data)

api_base = creds["anyscale_base_url"]
api_key = creds["anyscale_api_key"]
url = f"{api_base}/chat/completions"

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

s = requests.Session()

# News Extractor

news_extractor_system_message = """
You are tasked to extract key information from news articles.
You will be presented a news article that begins with ###News Article.

Instructions:
Extract the following items from the news article in the input in a JSON format:
{
    news setting: <Where did this news event take place?>,
    news characters and their main activities: <List names of up to five main \
stakeholders in this news event and what they mainly did.>
    news plot summary: <What happened in the news event?>
    news information points: <What are the three most important things in this news story?>
    news plot elements: <What are the four main plot points of the news story?>
}
To reiterate, your answer should be in the JSON format specified above.
"""

news_user_message_template = """###News Article \n{news_article}"""

news_article_input = """
The US Food and Drug Administration (USFDA) last week approved two breakthrough gene therapies — Casgevy by Vertex Pharmaceuticals and CRISPR Therapeutics, and Lyfgenia by Bluebird Bio — for sickle cell disease (SCD) in patients 12 years and older.

The development marks a milestone medical advancement in treating a debilitating disease that primarily affects red blood cells’ capacity to carry adequate oxygen across the body, with the use of innovative cell-based gene therapies.
Both approved products are made from patients’ own blood stem cells, which are modified, and are given back as a one-time, single-dose infusion as part of a hematopoietic (blood) stem cell transplant.

Casgevy utilises CRISPR/Cas9 (Clustered Regularly Interspaced Short Palindromic Repeats-CRISPR associated) technology, a type of genome editing system.
Emmanuelle Charpentier and Jennifer Doudna were awarded the Nobel Prize in Chemistry in 2020 for discovering CRISPR/Cas9 genetic scissors, called one of the gene technology’s sharpest tools.

In India, which has the highest number of SCD carriers in the world, scientists associated with the Council for Scientific and Industrial Research-Institute of Genomics and Integrative Biology (CSIR-IGIB) have been working since 2018 to develop a gene therapy for SCD using the same technology.

“After showing proof of the therapy developed in human-induced pluripotent stem cells (a particular potent type of stem cell that normally only exists during early embryonic development), we are now in preclinical stage of the therapy’s trial,” Debojyoti Chakraborty, lead scientist of the project at CSIR-IGIB, told ThePrint.

The next step after the animal study, he said, is to start a phase-1 clinical trial for SCD patients in India, in partnership with the All India Institute of Medical Sciences (AIIMS) in Delhi and the department of science and technology after regulatory approvals for the therapy are in place.
Once available in the country, the therapy can be a boon for millions of SCD patients in India which this year saw the launch of the National Sickle Cell Anaemia Elimination Mission — targeting to eliminate the disease by 2047.
"""

prompt_for_news_extraction = [
    {"role": "system", "content": news_extractor_system_message},
    {"role": "user", "content": news_user_message_template.format(
        news_article=news_article_input
        )
    }
]

body = {
  "model": model_name,
  "messages": prompt_for_news_extraction,
  "temperature": 0,
  "max_tokens": 8192
}

with s.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=body) as resp:
    news_extraction_output = json.loads(resp.json()['choices'][0]['message']['content'])

# Comedic Analogy

comedic_analogy_prompt_template = """
1. List three unique comedic analogies for the situation in the following story:
{news_plot_summary}. Incorporate the following characters only: {news_characters_and_their_main_activities}.
2. Decide the main characters of the news event as two of the most dominant characters in the summary: {news_plot_summary}
3. To act out this analogous premise use the location mentioned here: {news_setting}

Return your output as a JSON with the three analogies as keys like so:
- Comedic Analogy 1: <first analogy>,
- Comedic Analogy 2: <second analogy>,
- Comedic Analogy 3: <third analogy>
"""

user_prompt_for_comedic_analogy = comedic_analogy_prompt_template.format(
    news_plot_summary=news_extraction_output['news plot summary'],
    news_characters_and_their_main_activities=news_extraction_output['news characters and their main activities'],
    news_setting=news_extraction_output['news setting']
)

prompt_for_comedic_analogy = [
    {
        "role": "system", 
        "content": "You are tasked to create three comedic analogies based on key information extracted from a news article."
    },
    {
        "role": "user", 
        "content": user_prompt_for_comedic_analogy
    }
]

body = {
  "model": model_name,
  "messages": prompt_for_comedic_analogy,
  "temperature": 0,
  "max_tokens": 8192
}

with s.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=body) as resp:
    comedic_analogies = json.loads(resp.json()["choices"][0]["message"]["content"])

# Create Script
    
comedic_script_prompt_template = """
Write a script for a comedy skit about: {script_plot}. Cover the following information: {news_information_points}.
The characters should be exactly: {news_characters_and_their_main_activities}.
It should be set in {news_setting}. It should be entertaining.
The dialogue should be colloquial and engaging. The dialogue should be 10 to 12 lines long.
Each line of dialogue should be short - less than 20 words. End it with a punchline.
"""

user_prompt_for_comedic_script = comedic_script_prompt_template.format(
    script_plot=comedic_analogies['Comedic Analogy 1'],
    news_information_points=news_extraction_output['news information points'],
    news_characters_and_their_main_activities=news_extraction_output['news characters and their main activities'],
    news_setting=news_extraction_output['news setting']
)

prompt_for_comedic_script = [
    {
        "role": "system", 
        "content": "You are tasked to create a comedic skit."
    },
    {
        "role": "user", 
        "content": user_prompt_for_comedic_script
    }
]

body = {
  "model": model_name,
  "messages": prompt_for_comedic_script,
  "temperature": 0,
  "max_tokens": 8192
}

with s.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=body) as resp:
    comedic_script = resp.json()['choices'][0]['message']['content']

output_file_name = model_name.replace('/', '-') + '_comedic-script.txt'

with open(output_file_name, 'w') as f:
    f.writelines(comedic_script)
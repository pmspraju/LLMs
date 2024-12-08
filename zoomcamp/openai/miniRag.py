#######################################################################
# Use minisearch as our RAG model and use openai to set the context   #
#######################################################################

import minisearch
import json
import os
import sys

def readjson(file):
    with open(file, 'rt') as f:
        data_raw = json.load(f)

    documents = []

    for course_dict in data_raw:
        for doc in course_dict['documents']:
            doc['course'] = course_dict['course']
            documents.append(doc)
    return documents

def fit_index(docs):
    index = minisearch.Index(
        text_fields=["question", "text", "section"],
        keyword_fields=["course"]
    )
    index.fit(docs)
    return index

def search(query):
    boost = {'question': 3.0, 'section': 0.5}

    results = index.search(
        query=query,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        boost_dict=boost,
        num_results=5
    )

    return results


def build_prompt(query, search_results):
    prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

    context = ""

    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def exportKey():
    path = r'/home/nachiketa/Documents/Keys/openai'
    path = os.path.join(path, 'key')
    with open(path, 'rt') as f:
        key = f.read().strip()
    os.environ['OPENAI_API_KEY'] = key


def llm(prompt):
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def rag(query):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer

# Load the data
data_path = r'/home/nachiketa/Documents/Workspaces/LLMs/zoomcamp/data'
file = os.path.join(data_path, 'documents.json')
docs = readjson(file)

# Index the data using sklear vectorizer
index = fit_index(docs)

# test th Search and prmopt template
#query = 'the course has already started, can I still enroll?'
#query = 'where are the main videos are stored'
#query = 'when do i get the certificate'
#results = search(query)
#prompt = build_prompt(query, results)
#print(prompt)

# Set up the Open ai API
exportKey()
from openai import OpenAI
client = OpenAI()

#query = 'how do I run kafka?'
query = 'the course has already started, can I still enroll?'
answer = rag(query)
print(answer)













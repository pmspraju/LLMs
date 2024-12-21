import os
import gc
import streamlit as st
import json
import pandas as pd
import openpyxl
from mixedPrecision import ragModel

def loadoc():
    # Load the data
    data_path = r'/home/nachiketa/Documents/Workspaces/LLMs/zoomcamp/data'
    file = os.path.join(data_path, 'oldlist.xlsx')

    # Load the Excel file
    wb = openpyxl.load_workbook(file)
    sheet = wb.active

    headers = ['section','question','text']

    data = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        row_dict = {}
        for header, value in zip(headers, row):
            #row_dict[header] = value
            row_dict['text'] = row[2]
            row_dict['section'] = row[0]
            row_dict['question'] = row[1]

        data.append(row_dict)

    json_dict = {"course": "data-engineering-zoomcamp", "documents": data}
    json_data = []
    json_data.append(json_dict)

    json_file = os.path.join(data_path, 'oc.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    return json_file

def setPaths():

    os.environ['HF_HOME'] = '/home/nachiketa/Documents/Workspaces/HF_cache/flant5'

    # Load the data
    data_path = r'/home/nachiketa/Documents/Workspaces/LLMs/zoomcamp/data'
    file = os.path.join(data_path, 'documents.json')

    output_file = os.path.join(data_path, 'oc.json')
    if not os.path.exists(output_file):
        file = loadoc()
    else:
        file = output_file

    return file

if __name__ == "__main__":

    # set the paths and environment variables
    file = setPaths()

    # clear the cache
    gc.collect()

    # Set the query
    # query = 'the course has already started, can I still enroll?'
    #query = 'where are the main videos are stored'
    # query = 'when do i get the certificate'
    # query = 'how do I run kafka?'

    # Get the answer
    #answer = ragModel(file, query).rag()
    #print(answer)

    # show on the ui
    st.title("RAG Function Invocation")
    user_input = st.text_input("Enter your input:")

    if st.button("Ask"):
        with st.spinner('Processing...'):
            output = ragModel(file, user_input, True).rag()
            st.success("Completed!")
            st.write(output)


# Open a terminal.
# Navigate to the directory where your Streamlit script is located.
# Run the following command:
#  streamlit run /home/nachiketa/Documents/Workspaces/LLMs/zoomcamp/openai/flant5.py
# Open a web browser and navigate to http://localhost:8501/
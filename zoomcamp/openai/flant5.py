import os
import gc
import streamlit as st
import time
from mixedPrecision import ragModel


def setPaths():

    os.environ['HF_HOME'] = '/home/nachiketa/Documents/Workspaces/HF_cache/flant5'

    # Load the data
    data_path = r'/home/nachiketa/Documents/Workspaces/LLMs/zoomcamp/data'
    file = os.path.join(data_path, 'documents.json')

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
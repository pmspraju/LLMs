import os
import gc
from mixedPrecision import ragModel

os.environ['HF_HOME'] = '/home/nachiketa/Documents/Workspaces/HF_cache/flant5'

# Load the data
data_path = r'/home/nachiketa/Documents/Workspaces/LLM/zoomcamp/data'
file = os.path.join(data_path, 'documents.json')

#query = 'the course has already started, can I still enroll?'
query = 'where are the main videos are stored'
#query = 'when do i get the certificate'
#query = 'how do I run kafka?'

gc.collect()
answer = ragModel(file, query).rag()
print(answer)



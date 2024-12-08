import os.path

def exportKey():
    path = r'/home/nachiketa/Documents/Keys/openai'
    path = os.path.join(path, 'key.txt')
    with open(path, 'rt') as f:
        key = f.read()
    print(key)

index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"}
        }
    }
}
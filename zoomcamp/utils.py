import os.path

def exportKey():
    path = r'/home/nachiketa/Documents/Keys/openai'
    path = os.path.join(path, 'key.txt')
    with open(path, 'rt') as f:
        key = f.read()
    print(key)

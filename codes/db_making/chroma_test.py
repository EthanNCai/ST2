from chromadb import PersistentClient,Settings
from chromadb.utils import embedding_functions
import hashlib
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

def sha256_str(str_in: str) -> str:
    return hashlib.md5(str_in.encode('utf-8')).hexdigest()

chunk_size = 200
chunk_overlap = 20

time_start_1 = time.time()

with open("hongloumeng.txt", "r",encoding='utf-8') as f:  # 打开文件
    raw_texts = f.read()  # 读取文件

text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
texts = text_splitter.split_text(raw_texts)

client = chromadb.Client(settings=Settings(chroma_server_host='localhost',
                                           chroma_server_http_port= 8899,
                                           allow_reset=True))
# client = PersistentClient(settings=Settings(persist_directory="./dbtest",allow_reset=True))

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='../../moka-ai/m3e-large', device='cuda')
client.reset()
collection = client.get_or_create_collection(name="a_nice_collection",
                                      embedding_function=embedding_function,)

ids = [sha256_str(text) for text in texts]
collection.add(
    documents=texts,
    ids=ids,
)

time_end_1 = time.time()

print(len(texts))
print("time" + str(time_end_1 - time_start_1) + "s")
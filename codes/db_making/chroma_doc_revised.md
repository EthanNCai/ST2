# 使用Chromadb创建和维护向量数据库


包含的内容：

* chromadb的两个概念： Client 和 Collection
* 如何增\删\改\查\维护向量数据库
* 部署chromadb server

 
## 1 安装

```
pip install chromadb
```

chromadb 依托于 sqlite，如果sqlite版本太低可能会被提示更新sqlite

## 2 介绍

Chromadb结构很简单，它有两个概念
* Client __(is the object that wraps a connection to a backing Chroma DB)__
* Collection __(is the object that wraps a collection)__


Client的功能

* 管理（增删改查）Collection
* 负责持久化数据库

Collection的功能

* 增删改查向量

## 3 生成管理数据库


### 3.1 创建Client

创建client的时候可以传入一个settings，这个settings直接决定了client的性质，下面介绍两种不同的client

#### 3.1.1 其一本地化储存的数据库
 
通过下面的代码来创建一个可被持久化的数据库，运行完下面的代码就会在本地创建一个位于./dbtest的sqlite数据库。

```python
client = Client(settings=Settings(persist_directory="./dbtest",allow_reset=True))
```

#### 3.1.2 其二HttpClient

通过下面的代码来创建一个访问Chromadb服务器的Client，运行下面的代码就会创建一个连接本地端口在8899的Chroma服务器的客户端。

```python
client = chromadb.Client(settings=Settings(chroma_server_host='localhost', 
                                           chroma_server_http_port= 8899,
                                           allow_reset=True))
```

在使用这种Client之前，我们需要利用docker创建一个chromadb服务器，执行下面的指令就可以在8899端口开一个chromadb

```
docker pull chromadb/chroma
docker run -d --name chromadb-container -p 8899:8000 chromadb/chroma
```

chromadb的数据库会保存到容器里面，可以通过这个指令进入这个容器查看

```
docker exec -it chromadb-container /bin/bash
```


除了上面介绍的两种Client之外，chromdb还有多种Client的变种用于满足不同需求：[文档在此](https://docs.trychroma.com/reference/py-client)，比如说有一种Client会创建存在内存中的数据库用于测试，这种数据库在关闭程序的时候就会消失。

### 3.2 在Client里创建和管理collection

有了Client，我们可以通过 `Client` 的 `get_or_create_collection` 方法来创建或者获得collection，也就是类似于数据表的东西


正常情况下，一个向量数据库表里面所有的文字记录的向量数据都应该用 __同一个__ 嵌入模型来生成，只有这样互相比较相似度才是有意义的，因此，chromadb创建collections的时候可以顺带传入一个embeddings_function，也就是将某个嵌入模型和一张数据表 "绑定"

下面的代码展示了创建Collections的过程。 这里我传入嵌入模型的形式是Sentence Transformer，而chromadb也支持多种嵌入模型形式：[文档在此](https://docs.trychroma.com/guides/embeddings)

```python
embedding_fn = 
embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='/home/cjz/models/m3e-large'
    )
collection = client.get_or_create_collection(
    name="collection_name",
    embedding_function=embedding_fn
    )
```
其实，不止嵌入模型和collection是一一对应的关系，还有距离计算函数等，并且它也可以作为参数指定：[文档在此](https://docs.trychroma.com/guides)，除此之外，chromadb还提供有多种不同的collection的创建函数用于满足不同需求：[文档在此](https://docs.trychroma.com/reference/py-client)

当然Client可以创建不止一个Colletion，所以Client会有一些用于管理它们的函数，如下所示：

```python
.count_collections()
.list_collections()
.get_settings()
.get_version()
``` 

### 3.3 对Colletion进行的增删改查

现在我们已经获得了一个空的Collect
前面说过collection 是一个数据表，每一条记录都有几个字段

* document 用于储存原文
* metadata 用于储存一些自定义的参数，查询和删除的时候可以用里面的参数来筛选结果
* id 用于唯一标识一段原文
* embeddings 原文的嵌入向量

Collection有一些自带的很好用的函数：

```python
.peek() # returns a list of the first 10 items in the collection
.count() # returns the number of items in the collection
.json # json format
.dict # python dict format
```



这是我们的实验数据，假设这就是我们已经清理分段好的一些文本:

```python
texts = ["挺好的，我感觉这个很棒",
         "不错的，这个确实很好，我很推荐",
         "这玩意太垃圾了把",
         "这个东西真的是十分的糟糕", 
         "很好，我喜欢这个东西"]
```

### 3.3.1 增加数据

我们可以方便的用collection的add函数来往collection里面添加记录，就像下面这样，值得注意的是，我们不用手动传入embeddings这个字段，chromadb会自动将documents用我们之前创建collection时传入的嵌入函数来生成document对应的embeddings字段的值。

```python
collection.add(
    documents=texts,
    ids=[sha256_str(text) for text in texts]
    # 把每个文本的sha256值作id
)
```

### 3.3.2 查找

chromadb支持两种查找方式

* query相似度查找： 输入一段文本，或者向量，输出相似性高的记录（主要的查询方法）
* get精确查找：输入id或metadata的过滤条件，输出符合过滤条件的记录（本文暂不介绍这个：[文档在此](https://docs.trychroma.com/guides)）

这是query的例子，传入需要查找的数量以及query的文本，注意：返回的结果数量实际上是 query_tests的数量 乘以 n_results

example 1

```python
# example1
result = collection.query(
    query_texts=["不错，我觉得挺好","好棒啊"],
    n_results=2,
)
print(result['documents'])
print(len(result['documents']))

```

```python
{'ids': [['64fe45e91c10ed677346034db5ca5b8d'], ['64fe45e91c10ed677346034db5ca5b8d']],
 'distances': [[178.84136962890625], [483.3851013183594]], 
 'metadatas': [[{'date': '2021-01-02'}], [{'date': '2021-01-02'}]], 
 'embeddings': None, 
 'documents': [['挺好的，我感觉这个很棒'], ['挺好的，我感觉这个很棒']], 
 'uris': None, 
 'data': None}
```

example 2

```python
# example2
result = collection.query(
    query_texts=["这真的很糟糕"],
    n_results=2,
)
print(result['documents'])
print(len(result['documents']))
```


```python
{'ids': [['64fe45e91c10ed677346034db5ca5b8d'], ['64fe45e91c10ed677346034db5ca5b8d']],
 'distances': [[178.84136962890625], [483.3851013183594]],
  'metadatas': [[{'date': '2021-01-02'}], [{'date': '2021-01-02'}]], 'embeddings': None, 
  'documents': [['挺好的，我感觉这个很棒'], ['挺好的，我感觉这个很棒']], 
  'uris': None, 
  'data': None}
```

### 3.3.3 修改

我们可以根据id来更改记录，也可以使用每条数据中的metadata记录的东西来过滤出需要更新的目标。

对于update，如果只是传入新的document，那么chromadb做的就是简单的根据新的document重新计算这个id的的embeddings

```python
collection.update(
    documents=texts,
    ids=[sha256_str(text) for text in texts]
)
```

上面的update函数在查不到传入的id的时候将会报错，下面的upsert函数则可以在查不到id的时候创建一个新的记录
```python
new_text = ['我觉得不好不坏','一般般把']
collection.upsert(
    documents=new_text,
    ids=[sha256_str(text) for text in new_text]
)
```

### 3.3.4 根据文本删除对应的向量记录

<!-- 我们可以根据id来删除记录，同样也可以使用记录中的metadata来过滤出需要删除的目标 -->

```python
del_target_document = ['我觉得不好不坏']
collection.delete(
    ids=[sha256_str(text) for text in del_target_document]
)
```

## 4 插入性能测试

将整本《红楼梦》的文本以下面的参数切割

chunk_size = 200
chunk_overlap = 20

使用下面的代码生成向量数据库

```python
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

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='/home/cjz/models/m3e-large',device='cuda')
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

```

Chromadb - 5896 个分段 - 每个分段 200个token左右

m3e-large
显存 2500 MiB 
生成时间 25.01 S 
数据库大小 82.1MB

bert-base-chinese:
显存 1100 MiB 
生成时间 10.90 S 
数据库大小 58.2MB

## 5 参考文献

*   [Chroma 官方文档-getting started](https://docs.trychroma.com/getting-started)
*   [Chroma 官方文档-云服务器部署](https://docs.trychroma.com/deployment/aws)
*   [Chroma 官方文档-Collection和Client](https://docs.trychroma.com/reference/py-client)
*   [Chroma向量数据库chromadb](https://www.jianshu.com/p/9cc719d555b1)
*   [docker-compose启动项目时报错Version in “./docker-compose.yml“ is unsupported.](https://blog.csdn.net/qq_35716085/article/details/135065241)



## 6 完整Demo

```python
from chromadb import PersistentClient,Settings
from chromadb.utils import embedding_functions
import hashlib
import chromadb

def sha256_str(str_in: str) -> str:
    return hashlib.md5(str_in.encode('utf-8')).hexdigest()

source_texts = [("挺好的，我感觉这个很棒","2021-01-02"),
         ("不错的，这个确实很好，我很推荐","2021-01-03"),
         ("这玩意太垃圾了把","2021-01-04"),
         ("这个东西真的是十分的糟糕","2021-01-05"),
         ("很好，我喜欢这个东西","2021-01-06")]


# 创建数据库
client = chromadb.Client(settings=Settings(chroma_server_host='localhost',
                                           chroma_server_http_port= 8899,
                                           allow_reset=True))
# client = PersistentClient(settings=Settings(persist_directory="./dbtest",allow_reset=True))
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='/home/cjz/models/m3e-large')

client.reset()
collection = client.get_or_create_collection(name="a_nice_collection",
                                      embedding_function=embedding_function,)


# 增加数据

texts = [text for text,_ in source_texts]
dates = [{"date":date} for _, date in source_texts]
ids = [sha256_str(text) for text in texts]

# print(dates)

collection.add(
    documents=texts,
    ids=ids,
    metadatas=dates
)



# 查数据
result = collection.query(
    query_texts=["不错，我觉得挺好","好棒啊"],
    n_results=1,
)
print(result)

# 更新
# collection.update(
#     documents=texts,
#     ids=[sha256_str(text) for text in texts]
# )

new_text = ['我觉得不好不坏','一般般把']
collection.upsert(
    documents=new_text,
    ids=[sha256_str(text) for text in new_text]
)

print(result)


# 删除
del_target_document = ['我觉得不好不坏']
collection.delete(
    ids=[sha256_str(text) for text in del_target_document]
)


print(collection.peek()['documents'])
#


```
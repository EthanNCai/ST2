import numpy as np
from numpy.linalg import norm

A = np.array([2,1,2])
B = np.array([3,4,2])

cosine = np.dot(A,B)/(norm(A)*norm(B))
print("余弦相似度:", cosine)


A = np.array([20,10,20])
B = np.array([30,40,20])

cosine = np.dot(A,B)/(norm(A)*norm(B))
print("余弦相似度:", cosine)


A = np.array([2000,-10,-20])
B = np.array([3000,10,20])

cosine = np.dot(A,B)/(norm(A)*norm(B))
print("余弦相似度:", cosine)

A = np.array([2000,-1000,-2000])
B = np.array([3000,1000,2000])

cosine = np.dot(A,B)/(norm(A)*norm(B))
print("余弦相似度:", cosine)


A = np.array([2000,-1000,2000])
B = np.array([3000,1000,2000])

cosine = np.dot(A,B)/(norm(A)*norm(B))
print("余弦相似度:", cosine)
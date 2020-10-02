import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

G = nx.Graph()

G.add_nodes_from([1,2,3,4,5])
G.add_edges_from([(1, 2), (1, 3), (2, 1), (2,3), (2,4), (2,5),(3,1),(3,2),(3,4),(4,3),(4,2),(4,5),(5,2),(5,4)])
# print(G.nodes)
L = nx.linalg.laplacianmatrix.laplacian_matrix(G)
L = L.todense()
eigen_values, eigen_vectors = np.linalg.eigh(L)
f = np.matrix([[3],[4],[1],[6],[2]])
g = np.matrix([[4], [6], [7], [8], [1]])
Fg = np.matmul(np.matrix(eigen_vectors.T), g)
Ff = np.matmul(np.matrix(eigen_vectors.T), f)

print(Ff)
print(Fg)

# eigen_vectors = np.array(eigen_vectors)

# for i in range(len(eigen_vectors)):
#     for j in range(len(eigen_vectors[0])):
#         print("{0:.{1}f}".format(float(eigen_vectors[i][j]), 5), end=" & ")
#     print()


# res = []
# for i in range(len(Fg)):
#     res.append([float(Fg[i][0]*Ff[i][0])])

# print(np.matmul(np.matrix(eigen_vectors), np.matrix(res)))

# print(L.todense())
# vec = nx.laplacian_vectors
# A = nx.linalg.adjacency_matrix(G)
# f = np.matrix([[3],[4],[1],[6],[2]])
# print(A.todense())
# print(np.matmul(A.todense(), f))
import networkx as nx
import numpy.linalg 
import numpy as np

Nbound=10 #we choose a uniform bound for sampling number of edges and vertices
num_iter=10 #just how many random graphs you want
graph_spec_detail=[] #this will be the list of elements '[w,z]' where w is the graph and z is the assoc spectra

def graph_eigens(graph):
  '''
  The input is the graph, and the output is a normalized spectrum of the graph using the functions on 'Networkx'.
  '''
  return nx.normalized_laplacian_spectrum(graph)

def graph_spec_generator(Nbound, num_iter):
  '''
  The input here is the uniform sampling bound introduced above, and the number of sampling iterations. For each iteration, we choose
  an m and an n, the size and order, then we generate a random graph. Then we use the 'graph_eigens' to compute the spectra for this graph, and we 
  store the data in graph_spec_detail in the form '[w,z]' as mentioned above.
  '''
  for index in range(1,num_iter):
    xy=np.random.randint(1,Nbound,size=2)
    g=nx.gnm_random_graph(xy[0],xy[1])
    graph_spec_detail.append([g,graph_eigens(g)])
    
  return graph_spec_detail
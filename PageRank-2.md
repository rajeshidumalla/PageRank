# CS246 - Colab 5
## PageRank

### Setup

First of all, we authenticate a Google Drive client to download the dataset we will be processing in this Colab.

**Make sure to follow the interactive instructions.**


```python
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-1-7f0ea5eddfa2> in <module>
          1 from pydrive.auth import GoogleAuth
          2 from pydrive.drive import GoogleDrive
    ----> 3 from google.colab import auth
          4 from oauth2client.client import GoogleCredentials
          5 


    ModuleNotFoundError: No module named 'google.colab'



```python
id='1EoolSK32_U74I4FeLox88iuUB_SUUYsI'
downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('web-Stanford.txt')
```

If you executed the cells above, you should be able to see the dataset we will use for this Colab under the "Files" tab on the left panel.

Next, we import some of the common libraries needed for our task.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

### Data Loading

For this Colab we will be using [NetworkX](https://networkx.github.io), a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

The dataset we will analyze is a snapshot of the Web Graph centered around [stanford.edu](https://stanford.edu), collected in 2002. Nodes represent pages from Stanford University (stanford.edu) and directed edges represent hyperlinks between them. [[More Info]](http://snap.stanford.edu/data/web-Stanford.html)


```python
import networkx as nx

G = nx.read_edgelist('web-Stanford.txt', create_using=nx.DiGraph)
```


```python
print(nx.info(G))
```

    Name: 
    Type: DiGraph
    Number of nodes: 281903
    Number of edges: 2312497
    Average in degree:   8.2032
    Average out degree:   8.2032


### Your Task

To begin with, let's simplify our analysis by ignoring the dangling nodes and the disconnected components in the original graph.

Use NetworkX to identify the **largest** weakly connected component in the ```G``` graph.  From now on, use this connected component for all the following tasks.


```python
# YOUR CODE HERE
cc = max(nx.weakly_connected_components(G), key=len)
len(cc)
```




    255265




```python
to_remove = set(G.nodes()) - cc
```


```python
G.number_of_nodes()
```




    281903




```python
G.remove_nodes_from(to_remove)
```


```python
G.number_of_nodes()
```




    255265



Compute the PageRank vector, using the default parameters in NetworkX: [https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html#networkx.algorithms.link_analysis.pagerank_alg.pageranky](https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html#networkx.algorithms.link_analysis.pagerank_alg.pagerank)


```python
# YOUR CODE HERE
pr = nx.pagerank(G)
```


```python
len(pr)
```




    255265



In 1999, Barabási and Albert proposed an elegant mathematical model which can generate graphs with topological properties similar to the Web Graph (also called Scale-free Networks).

If you complete the steps below, you should obtain some empirical evidence that the Random Graph model is inferior compared to the Barabási–Albert model when it comes to generating a graph resembling the World Wide Web!

As such, we will use two different graph generator methods, and then we will test how well they approximate the Web Graph structure by means of comparing the respective PageRank vectors. [[NetworkX Graph generators]](https://networkx.github.io/documentation/stable/reference/generators.html#)

Using for both methods ```seed = 1```, generate:


1.   a random graph (with the fast method), setting ```n``` equal to the number of nodes in the original connected component, and ```p = 0.00008```
2.   a Barabasi-Albert graph (with the standard method), setting ```n``` equal to the number of nodes in the original connected component, and finding the right ***integer*** value for ```m``` such as the resulting number of edges **approximates by excess** the number of edges in the original connected component

and compute the PageRank vectors for both graphs.



```python
# YOUR CODE HERE
random_G = nx.fast_gnp_random_graph(n=G.number_of_nodes(), p=0.00008)
BA_G = nx.barabasi_albert_graph(n=G.number_of_nodes(), m=G.number_of_edges() // G.number_of_nodes()+1)
```


```python
print(G.number_of_edges())
print(random_G.number_of_edges())
print(BA_G.number_of_edges())
```

    2234572
    2606241
    2297304


Compare the PageRank vectors obtained on the generated graphs with the PageRank vector you computed on the original connected component.
**Sort** the components of each vector by value, and use cosine similarity as similarity measure. 

Feel free to use any implementation of the cosine similarity available in third-party libraries, or implement your own with ```numpy```.


```python
# YOUR CODE HERE
pr_random = nx.pagerank(random_G)
pr_ba = nx.pagerank(BA_G)
```


```python
sorted_pr = sorted(pr.values())
sorted_pr_random = sorted(pr_random.values())
sorted_pr_ba = sorted(pr_ba.values())
```


```python
from numpy.linalg import norm

a = np.array(sorted_pr)
b = np.array(sorted_pr_random)
cos_sim_random = a @ b.T /(norm(a)*norm(b))
print(cos_sim_random)

c = np.array(sorted_pr_ba)
cos_sim_ba = a @ c.T /(norm(a)*norm(c))
print(cos_sim_ba)
```

    0.10490560362519201
    0.677749779847275


Once you have working code for each cell above, **head over to Gradescope, read carefully the questions, and submit your solution for this Colab**!

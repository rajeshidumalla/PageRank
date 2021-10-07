# PageRank

PageRank (PR) is an algorithm used by Google Search to rank web pages in their search engine results. It is named after both the term "web page" and co-founder Larry Page. PageRank is a way of measuring the importance of website pages. According to Google:

PageRank works by counting the number and quality of links to a page to determine a rough estimate of how important the website is. The underlying assumption is that more important websites are likely to receive more links from other websites.

Currently, PageRank is not the only algorithm used by Google to order search results, but it is the first algorithm that was used by the company, and it is the best known.

### Setup

First of all, we authenticate a Google Drive client to download the dataset we will be processing in this Colab.


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


```python
id='1EoolSK32_U74I4FeLox88iuUB_SUUYsI'
downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('web-Stanford.txt')
```

If you executed the cells above, you should be able to see the dataset we will use for this Colab under the "Files" tab on the left panel.

Next, I import some of the common libraries needed for our task.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

### Data Loading

For this one, I will be using [NetworkX](https://networkx.github.io), a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

The dataset that I am going to analysis is a snapshot of the Web Graph centered around [stanford.edu](https://stanford.edu), collected in 2002. Nodes represent pages from Stanford University (stanford.edu) and directed edges represent hyperlinks between them. [[More Info]](http://snap.stanford.edu/data/web-Stanford.html)


```python
import networkx as nx

G = nx.read_edgelist('web-Stanford.txt', create_using=nx.DiGraph)
```


```python
print(nx.info(G))
```

    DiGraph with 281903 nodes and 2312497 edges


### Building PageRank

To begin with, let's simplify the analysis by ignoring the dangling nodes and the disconnected components in the original graph.

I am using NetworkX to identify the **largest** weakly connected component in the ```G``` graph.


```python
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



Computing the PageRank vector, using the default parameters in NetworkX: [https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html#networkx.algorithms.link_analysis.pagerank_alg.pageranky](https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html#networkx.algorithms.link_analysis.pagerank_alg.pagerank)


```python
pr = nx.pagerank(G)
```


```python
len(pr)
```




    255265



In 1999, Barabási and Albert proposed an elegant mathematical model which can generate graphs with topological properties similar to the Web Graph (also called Scale-free Networks).

By completing the steps below,  I will obtain some empirical evidence that the Random Graph model is inferior compared to the Barabási–Albert model when it comes to generating a graph resembling the World Wide Web!

As such, I am going to use two different graph generator methods, and then I will test how well they approximate the Web Graph structure by means of comparing the respective PageRank vectors. [[NetworkX Graph generators]](https://networkx.github.io/documentation/stable/reference/generators.html#)

Using for both methods ```seed = 1```, generate:


1.   a random graph (with the fast method), setting ```n``` equal to the number of nodes in the original connected component, and ```p = 0.00008```
2.   a Barabasi-Albert graph (with the standard method), setting ```n``` equal to the number of nodes in the original connected component, and finding the right ***integer*** value for ```m``` such as the resulting number of edges **approximates by excess** the number of edges in the original connected component and will compute the PageRank vectors for both graphs.



```python
random_G = nx.fast_gnp_random_graph(n=G.number_of_nodes(), p=0.00008)
BA_G = nx.barabasi_albert_graph(n=G.number_of_nodes(), m=G.number_of_edges() // G.number_of_nodes()+1)
```


```python
print(G.number_of_edges())
print(random_G.number_of_edges())
print(BA_G.number_of_edges())
```

    2234572
    2607875
    2297304


Then, I am going to compare the PageRank vectors obtained on the generated graphs with the PageRank vector that I have computed on the original connected component.

After that, I will **Sort** the components of each vector by value, and will use cosine similarity as similarity measure.


```python
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

    0.10475231269645234
    0.6600519707372204


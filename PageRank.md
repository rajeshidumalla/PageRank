{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "PageRank.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "toc_visible": true,
      "machine_shape": "hm"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kPt5q27L5557"
      },
      "source": [
        "# PageRank"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "p0-YhEpP_Ds-"
      },
      "source": [
        "### Setup"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "PUUjUvXe3Sjk"
      },
      "source": [
        "First of all, we authenticate a Google Drive client to download the dataset we will be processing in this Colab."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "lRElWs_x2mGh"
      },
      "source": [
        "from pydrive.auth import GoogleAuth\n",
        "from pydrive.drive import GoogleDrive\n",
        "from google.colab import auth\n",
        "from oauth2client.client import GoogleCredentials\n",
        "\n",
        "# Authenticate and create the PyDrive client\n",
        "auth.authenticate_user()\n",
        "gauth = GoogleAuth()\n",
        "gauth.credentials = GoogleCredentials.get_application_default()\n",
        "drive = GoogleDrive(gauth)"
      ],
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "QHsFTGUy2n1c"
      },
      "source": [
        "id='1EoolSK32_U74I4FeLox88iuUB_SUUYsI'\n",
        "downloaded = drive.CreateFile({'id': id})\n",
        "downloaded.GetContentFile('web-Stanford.txt')"
      ],
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "qwtlO4_m_LbQ"
      },
      "source": [
        "If you executed the cells above, you should be able to see the dataset we will use for this Colab under the \"Files\" tab on the left panel.\n",
        "\n",
        "Next, I import some of the common libraries needed for our task."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "twk-K-jilWK7"
      },
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "%matplotlib inline"
      ],
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kAYRX2PMm0L6"
      },
      "source": [
        "### Data Loading"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "GXzc_R6ArXtL"
      },
      "source": [
        "For this one, I will be using [NetworkX](https://networkx.github.io), a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.\n",
        "\n",
        "The dataset that I am going to analysis is a snapshot of the Web Graph centered around [stanford.edu](https://stanford.edu), collected in 2002. Nodes represent pages from Stanford University (stanford.edu) and directed edges represent hyperlinks between them. [[More Info]](http://snap.stanford.edu/data/web-Stanford.html)"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LPIadGxvLyyq"
      },
      "source": [
        "import networkx as nx\n",
        "\n",
        "G = nx.read_edgelist('web-Stanford.txt', create_using=nx.DiGraph)"
      ],
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Smd1XvR7MLyE",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "2fb73bf8-fdae-46c5-a92f-1223504fa46a"
      },
      "source": [
        "print(nx.info(G))"
      ],
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "DiGraph with 281903 nodes and 2312497 edges\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "vbmr23B2rJKR"
      },
      "source": [
        "### Building PageRank"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "x15OQeyys1xd"
      },
      "source": [
        "To begin with, let's simplify the analysis by ignoring the dangling nodes and the disconnected components in the original graph.\n",
        "\n",
        "I am using NetworkX to identify the **largest** weakly connected component in the ```G``` graph."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "R9tDwRidIw-Q",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d2809209-2107-4127-8214-a8bb36ba09ff"
      },
      "source": [
        "cc = max(nx.weakly_connected_components(G), key=len)\n",
        "len(cc)"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "255265"
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2E4FDLfc84bY"
      },
      "source": [
        "to_remove = set(G.nodes()) - cc"
      ],
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "K7ZxIrpn-D-8",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "318ecf8b-ba8c-4dfe-a2b1-b039170e9315"
      },
      "source": [
        "G.number_of_nodes()"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "281903"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "XSAZq2zm-EBK"
      },
      "source": [
        "G.remove_nodes_from(to_remove)"
      ],
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "aCbDVuk9-VdC",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "87d94d92-e775-499b-a8d9-9e8e3bd1656b"
      },
      "source": [
        "G.number_of_nodes()"
      ],
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "255265"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "mbYMNjBhuhK-"
      },
      "source": [
        "Computing the PageRank vector, using the default parameters in NetworkX: [https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html#networkx.algorithms.link_analysis.pagerank_alg.pageranky](https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html#networkx.algorithms.link_analysis.pagerank_alg.pagerank)"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ll-rVh7KVoLA"
      },
      "source": [
        "pr = nx.pagerank(G)"
      ],
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "UnZ5hTr-__LN",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "1776be71-f575-4140-c908-9292f6acac59"
      },
      "source": [
        "len(pr)"
      ],
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "255265"
            ]
          },
          "metadata": {},
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "xDx905Wk3FKf"
      },
      "source": [
        "In 1999, Barabási and Albert proposed an elegant mathematical model which can generate graphs with topological properties similar to the Web Graph (also called Scale-free Networks).\n",
        "\n",
        "By completing the steps below,  I will obtain some empirical evidence that the Random Graph model is inferior compared to the Barabási–Albert model when it comes to generating a graph resembling the World Wide Web!"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Ox3ksWEFyaP-"
      },
      "source": [
        "As such, I am going to use two different graph generator methods, and then I will test how well they approximate the Web Graph structure by means of comparing the respective PageRank vectors. [[NetworkX Graph generators]](https://networkx.github.io/documentation/stable/reference/generators.html#)\n",
        "\n",
        "Using for both methods ```seed = 1```, generate:\n",
        "\n",
        "\n",
        "1.   a random graph (with the fast method), setting ```n``` equal to the number of nodes in the original connected component, and ```p = 0.00008```\n",
        "2.   a Barabasi-Albert graph (with the standard method), setting ```n``` equal to the number of nodes in the original connected component, and finding the right ***integer*** value for ```m``` such as the resulting number of edges **approximates by excess** the number of edges in the original connected component and will compute the PageRank vectors for both graphs.\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "5Yd94CE9aPJP"
      },
      "source": [
        "random_G = nx.fast_gnp_random_graph(n=G.number_of_nodes(), p=0.00008)\n",
        "BA_G = nx.barabasi_albert_graph(n=G.number_of_nodes(), m=G.number_of_edges() // G.number_of_nodes()+1)"
      ],
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "TFj4Pm4TDd7L",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e58f25d4-1b88-4d9f-a3b8-f79d4c020766"
      },
      "source": [
        "print(G.number_of_edges())\n",
        "print(random_G.number_of_edges())\n",
        "print(BA_G.number_of_edges())"
      ],
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "2234572\n",
            "2607875\n",
            "2297304\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "BlxK42Pi01vN"
      },
      "source": [
        "Then, I am going to compare the PageRank vectors obtained on the generated graphs with the PageRank vector that I have computed on the original connected component.\n",
        "\n",
        "After that, I will **Sort** the components of each vector by value, and will use cosine similarity as similarity measure."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1aUgyeNdUQxs"
      },
      "source": [
        "pr_random = nx.pagerank(random_G)\n",
        "pr_ba = nx.pagerank(BA_G)"
      ],
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "9f-V3JIAETLE"
      },
      "source": [
        "sorted_pr = sorted(pr.values())\n",
        "sorted_pr_random = sorted(pr_random.values())\n",
        "sorted_pr_ba = sorted(pr_ba.values())"
      ],
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "QjfmQ-OYErNk",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ae7d535c-f653-4df5-c47f-674c92b5a1b5"
      },
      "source": [
        "from numpy.linalg import norm\n",
        "\n",
        "a = np.array(sorted_pr)\n",
        "b = np.array(sorted_pr_random)\n",
        "cos_sim_random = a @ b.T /(norm(a)*norm(b))\n",
        "print(cos_sim_random)\n",
        "\n",
        "c = np.array(sorted_pr_ba)\n",
        "cos_sim_ba = a @ c.T /(norm(a)*norm(c))\n",
        "print(cos_sim_ba)"
      ],
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0.10475231269645234\n",
            "0.6600519707372204\n"
          ]
        }
      ]
    }
  ]
}
import networkx as nx
import random
from gensim.models.word2vec import Word2Vec
import numpy as np

class Graph():
    def __init__(self, graph):
        self.g = graph

    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        """ Returns a truncated random walk.
            path_length: Length of the random walk.
            alpha: probability of restarts.
            start: the start node of the random walk.
        """

        if start:
            path = [start]
        else:
            # Sampling is uniform w.r.t V, and not w.r.t E
            pos = list(g.nodes())
            path = [rand.choice(pos)]

        while len(path) < path_length:
            cur = path[-1]

            nb_list = list(nx.neighbors(g, cur))

            if len(nb_list) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(nb_list))
                else:
                    path.append(path[0])
            else:
                break

        return [str(node) for node in path]

    def myrandom_walk1(self, path_length, alpha=0, rand=random.Random(), start=None):
        """ Returns a truncated random walk.
            path_length: Length of the random walk.
            alpha: probability of restarts.
            start: the start node of the random walk.
        """
        points = {node: 1.0 for node in self.g.nodes()}
        myratio = 1000.0

        if start:
            path = [start]
        else:
            # Sampling is uniform w.r.t V, and not w.r.t E
            pos = list(g.nodes())
            path = [rand.choice(pos)]

        points[path[0]] /= myratio

        while len(path) < path_length:
            cur = path[-1]

            nb_list = list(nx.neighbors(g, cur))

            if len(nb_list) > 0:
                if rand.random() >= alpha:
                    prob = [points[nb] for nb in nb_list]
                    path.append(np.random.choice(nb_list, p=prob/np.sum(prob)))
                    points[path[-1]] /= myratio
                else:
                    path.append(path[0])
            else:
                break

        return [str(node) for node in path]

    def perform_walks(self, number_of_walks, path_length):

        node_list = list(g.nodes())

        corpus = []
        for _ in range(number_of_walks):
            random.shuffle(node_list)
            for node in node_list:
                walk = self.myrandom_walk1(path_length=path_length, start=node)
                corpus.append(walk)

        self.corpus = corpus
        return corpus

    def save_corpus(self, filename):

        with open(filename, 'w') as f:
            for walk in self.corpus:
                line = "{}\n".format(" ".join(w for w in walk))
                f.write(line)


graph_path = "../datasets/citeseer.gml"

g = nx.read_gml(graph_path)

go = Graph(g)
corpus = go.perform_walks(number_of_walks=40, path_length=10)
#go.save_corpus(filename="./mywalk.corpus"

model = Word2Vec(corpus, sg=1, hs=1, workers=3, min_count=0, size=128)
model.wv.save_word2vec_format("mywalk.embedding")

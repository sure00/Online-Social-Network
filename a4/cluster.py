# You should not use any imports not listed here:
from collections import Counter, defaultdict, deque
import copy
import math
import networkx as nx
import urllib.request
import os
import pickle
import matplotlib.pyplot as plt

def loadData(filename):
    # The protocol version used is detected automatically, so we do not
    # have to specify it.

    if not os.path.isfile(filename):
        print("File %s do not exist, return derectly" %filename)
        return
    else:
        try:
            with open(filename, "rb") as file:
                unpickler = pickle.Unpickler(file)
                tweets = unpickler.load()
        except EOFError:
            return {}

    print(len(tweets))
    return tweets

def example_graph():
    """
    Create the example graph from class. Used for testing.
    Do not modify.
    """
    g = nx.Graph()
    g.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('D', 'E'), ('D', 'F'), ('D', 'G'), ('E', 'F'), ('G', 'F')])
    return g

def constructGraph(tweets):
    g = nx.Graph()
    for t in tweets:
        g.add_edges_from([(t['user']['id'], friend) for friend in t['user']['friends']])
    return g


def main():
    """
    FYI: This takes ~10-15 seconds to run on my laptop.
    """
    tweetFile = 'tweetsData.txt'
    tweets = loadData(tweetFile)
    graph = constructGraph(tweets)


    total = 0
    for t in tweets:
        #print("tweet id is %d, friends total have %d" %(t['user']['id'],len(t['user']['friends'])))
        total +=len(t['user']['friends'])

    print("total edge is", total)
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))


    list =[]

    for i in range(len(tweets)):
        for j in range(i+1,len(tweets)):

            list.append((tweets[i]['user']['id'], tweets[j]['user']['id'],
                         len(set(tweets[i]['user']['friends']) & set(tweets[j]['user']['friends']))))
            #print(set(users[i]['friends']) & set(users[j]['friends']))
    list = sorted(list , key=lambda  x:-x[2])

    print("list is", list)

    """
    graph = read_graph()
    print('graph has %d nodes and %d edges' %
          (graph.order(), graph.number_of_edges()))

    subgraph = get_subgraph(graph, 2)
    print('subgraph has %d nodes and %d edges' %
          (subgraph.order(), subgraph.number_of_edges()))
    print('norm_cut scores by max_depth:')
    print(score_max_depths(subgraph, range(1,5)))
    clusters = partition_girvan_newman(subgraph, 3)
    print('first partition: cluster 1 has %d nodes and cluster 2 has %d nodes' %
          (clusters[0].order(), clusters[1].order()))
    print('cluster 2 nodes:')
    print(clusters[1].nodes())

    test_node = 'Bill Gates'
    train_graph = make_training_graph(subgraph, test_node, 5)
    print('train_graph has %d nodes and %d edges' %
          (train_graph.order(), train_graph.number_of_edges()))


    jaccard_scores = jaccard(train_graph, test_node, 5)
    print('\ntop jaccard scores for Bill Gates:')
    print(jaccard_scores)
    print('jaccard accuracy=%g' %
          evaluate([x[0] for x in jaccard_scores], subgraph))

    path_scores = path_score(train_graph, test_node, k=5, beta=.1)
    print('\ntop path scores for Bill Gates for beta=.1:')
    print(path_scores)
    print('path accuracy for beta .1=%g' %
          evaluate([x[0] for x in path_scores], subgraph))
    """
if __name__ == '__main__':
    main()
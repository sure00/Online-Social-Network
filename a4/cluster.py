# You should not use any imports not listed here:
from collections import Counter, defaultdict, deque
import copy
import math
import networkx as nx
import urllib.request
import os
import pickle
import matplotlib.pyplot as plt
import itertools

clusterFile =  'cluster.pkl'
def loadData(filename):
    """ Load twittes which collected in collect period

    Args:
    None

    Returns:
    return twittes
    """
    if not os.path.isfile(filename):
        print("File %s do not exist, return derectly" %filename)
        return
    else:
        try:
            with open(filename, "rb") as file:
                unpickler = pickle.Unpickler(file)
                data = unpickler.load()
        except EOFError:
            return {}

    print("Load %d user data "%len(data))
    return data

def calcJaccard(user, k):
    scores=[]

    roots = user.keys()
    for pair in itertools.combinations(roots,2):
        set1=set(user[pair[0]])
        set2=set(user[pair[1]])

        scores.append(((pair[0], pair[1]), 1. * len(set1 & set2) / len(set1 | set2)))

    return sorted(scores, key=lambda x: (-x[1], x[0]))[0:k]
    #return sorted(scores, key=lambda x: (-x[1], x[0]))

def constructGraph(user, JaccardScore):
    """ Construct the Graph with tweets user id and his friends.

    Args:
    tweets: origial tweets data from tweets data

    Returns:
    return graph
    """
    rootlist=[]
    g = nx.Graph()

    for pair in JaccardScore:
        node1 = pair[0][0]
        node2 = pair[0][1]
        if node1 not in rootlist:
            #print("pair[0]",pair[0])
            #print("its friends is",user[node1] )
            g.add_edges_from([(node1, friend) for friend in user[node1]])
        if node2 not in rootlist:
            g.add_edges_from([(node2, friend) for friend in user[node2]])
        rootlist.extend(pair)
    return g

def friend_overlap(tweets):
    list =[]
    for i in range(len(tweets)):
        for j in range(i+1,len(tweets)):

            list.append((tweets[i]['user']['id'], tweets[j]['user']['id'],
                         len(set(tweets[i]['user']['friends']) & set(tweets[j]['user']['friends']))))
            #print(set(users[i]['friends']) & set(users[j]['friends']))
    list = sorted(list , key=lambda  x:-x[2])

    print("list is", list)


def find_best_edge(G0):
    eb = nx.edge_betweenness_centrality(G0)
    # eb is dict of (edge, score) pairs, where higher is better
    # Return the edge with the highest score.
    return sorted(eb.items(), key=lambda x: x[1], reverse=True)[0][0]

def girvan_newman(G, depth=0):
    """ Recursive implementation of the girvan_newman algorithm.
    See http://www-rohan.sdsu.edu/~gawron/python_for_ss/course_core/book_draft/Social_Networks/Networkx.html

    Args:
    G.....a networkx graph

    Returns:
    A list of all discovered communities,
    a list of lists of nodes. """

    if G.order() == 1:
        return [G.nodes()]

    # Each component is a separate community. We cluster each of these.
    components = [c for c in nx.connected_component_subgraphs(G)]
    indent = '   ' * depth  # for printing
    while len(components) == 1:
        edge_to_remove = find_best_edge(G)
        print(indent + 'removing ' + str(edge_to_remove))
        G.remove_edge(*edge_to_remove)
        components = [c for c in nx.connected_component_subgraphs(G)]

    print("len of compoents is", len(components))

    return components
    #result = [c.nodes() for c in components]
    #print(indent + 'components=' + str(result))
    #for c in components:
        #result.extend(girvan_newman(c, depth + 1))

    #return result

def draw_network(graph,  filename):
    """
    Draw the network to a file. Only label the candidate nodes; the friend
    nodes should have no labels (to reduce clutter).
    Methods you'll need include networkx.draw_networkx, plt.figure, and plt.savefig.
    Your figure does not have to look exactly the same as mine, but try to
    make it look presentable.
    """


    custom_node_sizes = {}

    #print("tweets is ", tweets)
    #for tweet in tweets:
        #custom_labels[tweet['user']['screen_name']] = tweet['user']['screen_name']

    #nx.draw(graph,  labels=custom_labels, node_list = custom_node_sizes.keys(), node_size=100,edge_color='c')
    nx.draw_networkx(graph, node_list = custom_node_sizes.keys(), node_size=100,edge_color='c',pos=nx.spring_layout(graph))
    plt.savefig(filename)

# append to database
def saveData(data,file):
           """ save the collect data to tweetsData.txt.
           Args:
             twitters .... Collect data from twitter.
           Returns:
             NULL
           """
           f = open(file, 'wb+')
           #tweets = [t for t in tweets if 'user' in t]
           #print('fetched %d tweets' % len(tweets))
           pickle.dump(data, f)
           f.close()
           print("Data %s saved successfully" %file)

def main():
    """
    FYI: This takes ~10-15 seconds to run on my laptop.
    """
    userFile = 'user.pkl'
    users = loadData(userFile)
    print("users is ", users)

    JaccardScore = calcJaccard(users, 3)
    print("Jaccard Score Top 3 is ", JaccardScore)

    graph = constructGraph(users, JaccardScore)
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))

    components = girvan_newman(graph)

    for i, c in enumerate(components):
        print("components have %s nodes" % (str(c.nodes())))
        draw_network(c, 'copmonent'+str(i)+'.png')
    draw_network(graph, 'network.png')

    saveData(components, clusterFile)
if __name__ == '__main__':
    main()
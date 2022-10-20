import random
import networkx as nx
import timeit
import numba as nb
import numpy
import matplotlib.pyplot as plt

#################################################################################
#################################################################################
# Code to plot S1 Fig
#################################################################################
#################################################################################


def giantComponentExists2(viables):
    G = nx.Graph()
    for key in viables:
        G.add_node(key)

    for n1 in G.nodes(data=True):
        for n2 in G.nodes(data=True):
            if bin(n1[0] ^ n2[0]).count("1") == 1:
                G.add_edge(n1[0], n2[0], weight=1)

    return nx.is_connected(G)


def giantComponentExists(nums):
    # Construct a graph as a dictionary
    graph = {n:[] for n in nums}

    # Add edges between nodes
    for n1 in nums:
        for n2 in nums:
            if bin(n1^n2).count("1") == 1:  #compare if binary rep. differs at one position
                graph[n1].append(n2)

    # BFS search to determine if graph is fully connected
    fringe = [min(nums)]
    visited = set()
    while fringe:
        for edge in graph[fringe[0]]:
            if edge not in visited:
                fringe += [edge]
        visited.add(fringe[0])
        fringe.pop(0)
    return len(visited) == len(graph.keys())


l=10
avg = 1000
datapoints = 30

p_array = numpy.linspace(0.0, 1.0, num=datapoints)
Werte = []

for p in p_array:
    print("p", p)
    counter = 0.0
    for i in range(avg):
        print("i", i)
        viables = set()
        lethals = set()
        connected = set()

        for x in range(2 ** l):
            if random.random() <= p:
                viables.add(x)
            else:
                lethals.add(x)
                6

        if len(viables) > 0:
            if giantComponentExists2(viables) or p>0.8:
                counter+=1

    Werte.append(counter/avg)

plt.figure(1, figsize=(4,3))
plt.title("$L={}$".format(l))
plt.plot(p_array, Werte)
plt.xlabel("$p$")
plt.ylabel("Prob. connected")
plt.tight_layout()
plt.show()



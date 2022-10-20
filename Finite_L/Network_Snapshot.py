import matplotlib.pyplot as plt
import timeit
import numba as nb
import random
import networkx as nx
from collections import Counter
import numpy
import matplotlib as mpl


#################################################################################
#################################################################################
# Code to generate data for Fig 13 & S.11.
#################################################################################
#################################################################################


mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


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
    #print("fringe", fringe)
    visited = set()
    while fringe:
        for edge in graph[fringe[0]]:
            if edge not in visited:
                fringe += [edge]
        visited.add(fringe[0])
        fringe.pop(0)
    return len(visited) == len(graph.keys())


@nb.njit
def evolution_selrec(generations, N, l, mu, r, p, viables, lethals):
    wp = [0] * N
    wp_temp2 = [0] * N

    rand_idx = random.randrange(0, len(viables))
    viable_first = list(viables)[rand_idx]

    for x3 in range(N):
        wp[x3] = viable_first
        wp_temp2[x3] = viable_first

    for _ in range(generations):

        if _ % 1000 == 0:
            print("gen", _)

        # selection + recombination
        for x3 in range(N):
            wp_temp2[x3] = wp[x3]
        for x3 in range(N):
            if random.random() <= r:
                randint1 = random.randint(0, N - 1)
                randint2 = random.randint(0, N - 1)
                while randint1 == randint2 or wp[randint1] in lethals or wp[randint2] in lethals:
                    randint1 = random.randint(0, N - 1)
                    randint2 = random.randint(0, N - 1)
                a = wp[randint1]
                b = wp[randint2]
                c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                wp_temp2[x3] = c
            else:
                randint1 = random.randint(0, N - 1)
                while wp[randint1] in lethals:
                    randint1 = random.randint(0, N - 1)
                c = wp[randint1]
                wp_temp2[x3] = c
        for x3 in range(N):
            wp[x3] = wp_temp2[x3]

        # mutation
        for mutation_in_individual in range(N):
            for mutation_at_loci in range(l):
                if random.random() < mu:
                    wp[mutation_in_individual] = wp[mutation_in_individual] ^ (1 << mutation_at_loci)

    return wp, viables


"""
mu = 0.0001  #0.0001   0.001
generations = 100000 #1000000   10000
r = 1.0
l = 14 #14  8
N = 10000 #10000 1000
p = 0.5
"""


mu = 0.0001  #0.0001   0.01
generations = 10000000 #10000000   1000000
r = 1.0
l = 14 #14  10
N = 10000 #10000 100
p = 0.7
landscape = "perc2"
save = True
load = False


#if load == True:
#    save = False


print("Generations", generations)
print("2**L=", 2**l)
print("r", r)
print("N*L*mu", N*l*mu)

start = timeit.default_timer()
if load is False:
    viables = set()
    lethals = set()
    ###############################
    ###percolation connected
    if landscape == "perc2":
        existsgiantcomp = False
        while existsgiantcomp == False:
            viables.clear()
            lethals.clear()
            for x in range(2 ** l):
                if random.random() <= p:
                    viables.add(x)
                else:
                    lethals.add(x)

            if giantComponentExists2(viables):
                existsgiantcomp = True

    ###############################
    ###percolation
    if landscape == "perc":
        for x in range(2 ** l):
            if random.random() <= p:
                viables.add(x)
            else:
                lethals.add(x)

    ###############################
    ###one_component
    if landscape == "gc":
        viables.add(0)
        ns = int(2 ** l * p)
        while len(viables) < ns:
            rand_idx = random.randrange(0, len(viables))
            choosen_viable = list(viables)[rand_idx]

            neighbors = []
            # del neighbors[:]
            for loci in range(l):
                neighbors.append(choosen_viable ^ (1 << loci))
            rand_idx = random.randrange(0, len(neighbors))
            viable_candidate = neighbors[rand_idx]
            viables.add(viable_candidate)

        for x in range(2 ** l):
            if x in viables:
                pass
            else:
                lethals.add(x)

print("landscape done")

if load is False:
    wp, viables = evolution_selrec(generations, N, l, mu, r, p, viables, lethals)
else:
    wp = numpy.load('Graph_data/Graph_wp_N{}_l{}_r{}_mu{}_gen{}_p{}_{}.npy'.format(N, l, r, mu, generations, p, landscape))
    viables = numpy.load('Graph_data/Graph_viables_N{}_l{}_r{}_mu{}_gen{}_p{}_{}.npy'.format(N, l, r, mu, generations, p, landscape))

stop = timeit.default_timer()
print("Time", stop-start)

if save is True:
    numpy.save('Graph_data/Graph_wp_N{}_l{}_r{}_mu{}_gen{}_p{}_{}.npy'.format(N, l, r, mu, generations, p, landscape), wp)
    numpy.save('Graph_data/Graph_viables_N{}_l{}_r{}_mu{}_gen{}_p{}_{}.npy'.format(N, l, r, mu, generations, p, landscape), list(viables))



uniques = Counter(wp)
print("uniques", uniques)
number_distinct_genotypes = len(uniques.keys())
print("number_distinct_genotypes", number_distinct_genotypes)
distinct_and_viable_genotypes = {x for x in uniques.keys() if x in viables}
print("number_distinct_and_viablesgenotypes",len(distinct_and_viable_genotypes))

#print("viables", viables)
zz1 = []
G = nx.Graph()
for key in uniques:
    if key in viables:
        ## add robustness label to node##
        neighbors = 0
        for x in range(2 ** l):
            if x in viables and bin(key ^ x).count("1") == 1:
                neighbors += 1
        m_sigma = 1.0 * neighbors / l
        G.add_node(key, frequency=uniques[key], m_sigma=m_sigma)
        #G.add_node(key, frequency=uniques[key])
        zz1.append(uniques[key])

if save is True:
    for n1 in G.nodes(data=True):
        for n2 in G.nodes(data=True):
            if bin(n1[0]^n2[0]).count("1") == 1:
                G.add_edge(n1[0], n2[0], weight=1)



    print("len(G.nodes", len(G.nodes))
    print("len(G.edges)", len(G.edges))
#print(zz1)
#pos = nx.spring_layout(G)
#pos = nx.kamada_kawai_layout(G)
#pos = nx.planar_layout(G)
#pos = nx.spectral_layout(G)

#plt.figure(1)
#nx.draw(G,pos, node_size=[20*item for item in zz1], with_labels=False)
if save is True:
    nx.write_gexf(G, "Graph_data/L{}_mu{}_r{}_p{}_{}.gexf".format(l, mu, r, p, landscape))

plt.figure(2, figsize=(3, 2))
print("zz1", zz1)
plt.bar(list(range(1, len(zz1)+1)), sorted(zz1)[::-1], width=1.0, color="black")
plt.xlim(1, len(zz1)+1)
#plt.xticks([1,50,100])
plt.semilogy()
if r == 1:
    plt.semilogx()

print("done")
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.tight_layout()
# Hide the right and top spines

plt.savefig('Hist_L{}_mu{}_r{}_p{}.png'.format(l, mu, r, p), transparent=True)


if p < 1.0:
    print("test")
    m = 0.0
    for key in uniques:
        if key in viables:
            neighbors = 0
            for x in range(2**l):
                if x in viables and bin(key^x).count("1")==1:
                    neighbors += 1
            m += 1.0*uniques[key]/N*neighbors/l
    print("m=", m)


plt.show()
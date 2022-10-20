import sys
sys.path.append("/home/klug/PycharmProjects/Finite_Population_Clean_TD/scr")
import numpy
import timeit
import numba as nb
import random
from modules import ham_dist_fct_nb

#################################################################################
#################################################################################
# Code to generate data for plot Fig. S8. Designed to be run from a shell script to parallize tasks.
#################################################################################
#################################################################################


@nb.njit
def neighbour_decimal_f(n, bitLength):
    neighbors = [0]
    del neighbors[:]
    for loci in range(bitLength):
        neighbors.append(n ^ (1 << loci))
    return neighbors


@nb.njit
def mutational_robustness_single_genotype(genotype, viables, l):
    robustness = 0
    neighbors = neighbour_decimal_f(genotype, l)
    for n in neighbors:
        if n in viables:
            robustness += 1
    robustness /= l
    return robustness


@nb.njit
def pfunc(r, mu, N, l, viables, lethals):

    wp = [0] * N
    wp_temp2 = [0] * N
    NL = N * l  # mutation v2 & v3
    #NL_range_tempalte = list(range(NL))  # mutation v3
    #selected = {0, 1, 2}  # mutation v3


    rand_idx = random.randrange(0, len(viables))
    viable_first = list(viables)[rand_idx]

    rand_idx = random.randrange(0, len(viables))
    escape_variant = list(viables)[rand_idx]

    robustness_i = mutational_robustness_single_genotype(viable_first, viables, l)
    robustness_e = mutational_robustness_single_genotype(escape_variant, viables, l)
    distance_i_e_variant =ham_dist_fct_nb(viable_first, escape_variant)

    for x3 in range(N):
        wp[x3] = viable_first
        wp_temp2[x3] = viable_first
    gen_counter = 0

    mut_counter_temp = 0

    escape_variant_found = False

    while escape_variant_found == False:
        gen_counter += 1

        segMut_temp = len(set(wp))
        # if not [wp[0]] * len(wp) == wp:
        if segMut_temp > 1:
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
            mut_counter_temp = 0
            for mutation_at_loci in range(l):
                if random.random() < mu:
                    mut_counter_temp += 1
                    wp[mutation_in_individual] = wp[mutation_in_individual] ^ (1 << mutation_at_loci)
            if wp[mutation_in_individual] == escape_variant:
                escape_variant_found = True


        #if flips > 0 or segMut_temp > 1 or gen_counter == 1:
        if mut_counter_temp > 0 or segMut_temp > 1 or gen_counter == 1:
            for x3 in range(N):
                if wp[x3] == escape_variant:
                    escape_variant_found = True


    returnarray = numpy.zeros(4)
    returnarray[0] = gen_counter
    returnarray[1] = distance_i_e_variant
    returnarray[2] = robustness_i
    returnarray[3] = robustness_e


    return returnarray


def giantComponentExists(nums):
    # Construct a graph as a dictionary
    graph = {n:[] for n in nums}
    #print(graph)

    # Add edges between nodes
    for n1 in nums:
        for n2 in nums:
            if bin(n1^n2).count("1") == 1:  #compare if binary rep. differs at one position
                graph[n1].append(n2)
    #print(graph)

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


def selrec_r_mu_cluster(r, mu, N, l, p, avg, landscape):

    gen_counter_list = []
    distance_i_e_variant_list = []
    robustness_i_list = []
    robustness_e_list = []

    viables = set()
    lethals = set()

    for _ in range(avg):

        print("avg", _)
        ###############################
        ###percolation
        if landscape == "perc":
            viables.clear()
            lethals.clear()
            for x in range(2 ** l):
                if random.random() <= p:
                    viables.add(x)
                else:
                    lethals.add(x)
        ###############################
        ###############################
        ###percolation
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

                if giantComponentExists(viables):
                    existsgiantcomp = True
        ###############################
        print("test")
        if p<1.0:
            returnarray = pfunc(r, mu, N, l, viables, lethals)


        gen_counter_list.append(returnarray[0])
        distance_i_e_variant_list.append(returnarray[1])
        robustness_i_list.append(returnarray[2])
        robustness_e_list.append(returnarray[3])

    return gen_counter_list, distance_i_e_variant_list, robustness_i_list, robustness_e_list


rmin = 0.001
rmax = 1.0
mu = 0.01
datapoints = 30
avg = 10000
log = True
l = 10
N = 100
p = 0.8
landscape = "perc"


x1 = int(sys.argv[1])



if log is False:
    rlist = numpy.linspace(rmin, rmax, datapoints)
if log is True:
    rlist_temp = numpy.linspace(numpy.log10(rmin), numpy.log10(rmax), datapoints)
    rlist = numpy.power(10, rlist_temp)


r = rlist[x1]


print("r=", r)
print("mu=", mu)



start = timeit.default_timer()
temp_values = selrec_r_mu_cluster(r, mu, N, l, p, avg, landscape)  #attention can also be sel_rec



nested_list = [r, temp_values[0], temp_values[1], temp_values[2], temp_values[3]]
print(nested_list)

numpy.save('Escape/L{}_N{}_p{}_avg{}_dp{}_i{}_{}.npy'.format(l, N, p, avg, datapoints, int(x1), landscape), nested_list)


stop = timeit.default_timer()
print("Time", stop-start)



f = open("timeit/Escape_L{}_N{}_p{}_avg{}.txt".format(l, N, p, avg), "a")
f.write("i{}: t={}s\n".format(x1, int(stop-start)))
f.close()


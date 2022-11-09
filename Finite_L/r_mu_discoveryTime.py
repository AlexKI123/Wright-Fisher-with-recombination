import sys
sys.path.append("/.../scr")
import numpy
import timeit
import numba as nb
import random
import math
from modules import mutational_robustness_nb_smr2_hd
from modules import ham_dist_fct_nb
import scipy.special


#################################################################################
#################################################################################
# Code to generate data for 3DPlot_r_mu_discoveryTime.py. Designed to be run from a shell script to parallize tasks.
#################################################################################
#################################################################################


@nb.njit
def pfunc(r, mu, N, l, viables, lethals, two_point_distances):
    gen_counter_avg = 0
    mut_counter_avg = 0
    m_value_avg = 0.0
    avg_hd_avg = 0.0
    nogenotypes_avg = 0.0
    nosegmut_avg = 0.0
    meanfitness_avg = 0.0

    wp = [0] * N
    wp_temp2 = [0] * N
    NL = N * l  # mutation v2 & v3
    NL_range_tempalte = list(range(NL))  # mutation v3
    selected = {0, 1, 2}  # mutation v3


    rand_idx = random.randrange(0, len(viables))
    viable_first = list(viables)[rand_idx]

    viables_copy = viables.copy()
    for x3 in range(N):
        wp[x3] = viable_first
        wp_temp2[x3] = viable_first
    gen_counter = 0
    mut_counter = 0
    missing_genotypes = 1

    m_value = 0.0
    avg_hd = 0.0
    nogenotypes = 0.0
    nosegmut = 0.0
    meanfitness = 0.0
    viable_rec_counter = 0
    recom_counter = 0

    mut_counter_temp = 0


    while missing_genotypes > 0:
        gen_counter += 1

        segMut_temp = len(set(wp))
        # if not [wp[0]] * len(wp) == wp:
        if segMut_temp > 1:
            # selection + recombination
            for x3 in range(N):
                wp_temp2[x3] = wp[x3]
            for x3 in range(N):
                if random.random() <= r:
                    recom_counter += 1
                    randint1 = random.randint(0, N - 1)
                    randint2 = random.randint(0, N - 1)
                    while randint1 == randint2 or wp[randint1] in lethals or wp[randint2] in lethals:
                        randint1 = random.randint(0, N - 1)
                        randint2 = random.randint(0, N - 1)
                    a = wp[randint1]
                    b = wp[randint2]
                    c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                    wp_temp2[x3] = c
                    if c in viables:
                        viable_rec_counter += 1
                else:
                    randint1 = random.randint(0, N - 1)
                    while wp[randint1] in lethals:
                        randint1 = random.randint(0, N - 1)
                    c = wp[randint1]
                    wp_temp2[x3] = c
            for x3 in range(N):
                wp[x3] = wp_temp2[x3]

        """
        # mutation
        for mutation_in_individual in range(N):
            mut_counter_temp = 0
            for mutation_at_loci in range(l):
                if random.random() < mu:
                    #mut_counter += 1
                    mut_counter_temp += 1
                    wp[mutation_in_individual] = wp[mutation_in_individual] ^ (1 << mutation_at_loci)
            if mut_counter_temp > 0:
                mut_counter += 1
            if wp[mutation_in_individual] in viables_copy:
                viables_copy.remove(wp[mutation_in_individual])
                missing_genotypes = len(viables_copy)
                if missing_genotypes == 0:
                    break

        """
        # mutation v3
        flips = numpy.random.binomial(NL, mu)
        if flips > 0:
            mut_counter += flips
            positions = [0] * flips
            setsize = 21  # size of a small set minus size of an empty list
            if flips > 5:
                setsize += 4 ** math.ceil(math.log(flips * 3) / math.log(4))  # table size for big sets
            if NL <= setsize:
                # An n-length list is smaller than a k-length set
                pool = NL_range_tempalte[:]
                for i in range(flips):  # invariant:  non-selected at [0,n-i)
                    j = random.randint(0, (NL - i) - 1)
                    positions[i] = pool[j]
                    pool[j] = pool[NL - i - 1]  # move non-selected item into vacancy
            else:
                # selected = set()
                selected.clear()
                selected_add = selected.add
                for i in range(flips):
                    j = random.randint(0, NL - 1)
                    while j in selected:
                        j = random.randint(0, NL - 1)
                    selected_add(j)
                    positions[i] = NL_range_tempalte[j]

            for pos in positions:
                # mut_counter += 1
                wp[pos // l] = wp[pos // l] ^ (1 << (pos % l))
                # if wp[pos // l] in viables_copy:
                #    viables_copy.remove(wp[pos // l])
                #    missing_genotypes = len(viables_copy)
                #    if missing_genotypes == 0:
                #        break


        if flips > 0 or segMut_temp > 1 or gen_counter == 1:
        #if mut_counter_temp > 0 or segMut_temp > 1 or gen_counter == 1:
            for x3 in range(N):
                if wp[x3] in viables_copy:
                    viables_copy.remove(wp[x3])
            missing_genotypes = len(viables_copy)

            temp_values = mutational_robustness_nb_smr2_hd(N, wp, viables, l, two_point_distances)

        m_value += 1.0 * temp_values[0]
        avg_hd += 1.0 * temp_values[1]
        nogenotypes += 1.0 * temp_values[2]
        nosegmut += 1.0 * temp_values[3]
        meanfitness += 1.0 * temp_values[4]


    gen_counter_avg += 1.0 * gen_counter
    mut_counter_avg += 1.0 * mut_counter
    m_value_avg += 1.0 * (m_value / gen_counter)
    avg_hd_avg += 1.0 * (avg_hd / gen_counter)
    nogenotypes_avg += 1.0 * (nogenotypes / gen_counter)
    nosegmut_avg += 1.0 * (nosegmut / gen_counter)
    meanfitness_avg += 1.0*(meanfitness/gen_counter)

    returnarray = numpy.zeros(8)
    returnarray[0] = gen_counter_avg
    returnarray[1] = mut_counter_avg
    returnarray[2] = m_value_avg
    returnarray[3] = avg_hd_avg
    returnarray[4] = nogenotypes_avg
    returnarray[5] = nosegmut_avg
    if recom_counter > 0:
        returnarray[6] = 1.0*viable_rec_counter/recom_counter
    else:
        returnarray[6] = 1.0
    returnarray[7] = meanfitness_avg

    return returnarray


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


def selrec_exploring_r_mu_cluster(r, mu, N, l, p, k, avg, landscape, two_point_distances):

    gen_counter_avg = 0
    mut_counter_avg = 0
    m_value_avg = 0.0
    avg_hd_avg = 0.0
    nogenotypes_avg = 0.0
    nosegmut_avg = 0.0
    rr_value_avg = 0.0
    meanFitness_value = 0.0
    viables = set()
    lethals = set()
    neighbors = [0]


    for _ in range(avg):
        viables.clear()
        lethals.clear()

        ###############################
        ###percolation
        if landscape == "perc":
            for x in range(2 ** l):
                if random.random() <= p:
                    viables.add(x)
                else:
                    lethals.add(x)
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
        ###one_component
        if landscape == "gc":
            viables.add(0)
            ns = int(2 ** l * p)
            while len(viables) < ns:
                rand_idx = random.randrange(0, len(viables))
                choosen_viable = list(viables)[rand_idx]

                del neighbors[:]
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

        ###############################
        returnarray = pfunc(r, mu, N, l, viables, lethals, two_point_distances)
        gen_counter_avg += returnarray[0] / avg
        mut_counter_avg += returnarray[1] / avg
        m_value_avg += returnarray[2] / avg
        avg_hd_avg += returnarray[3] / avg
        nogenotypes_avg += returnarray[4] / avg
        nosegmut_avg += returnarray[5] / avg
        rr_value_avg += returnarray[6] / avg
        meanFitness_value += returnarray[7] / avg

    return gen_counter_avg, mut_counter_avg, m_value_avg, avg_hd_avg, nogenotypes_avg, nosegmut_avg, rr_value_avg, meanFitness_value


rmin = 0.001
rmax = 1.0
mumin = 0.000001  # 0.0001   0.000001
mumax = 0.5
datapoints = 30  # 30
avg = 1000  # 1000 #10000
log = True
l = 10  # 6  10
N = 100
p = 0.5
k = 5
landscape = "perc"
mufliped= True



two_point_distances = scipy.special.binom(N, 2)

#x1 = int(sys.argv[1])
x1 = 0


if log is False:
    rlist = numpy.linspace(rmin, rmax, datapoints)
    mulist = numpy.linspace(mumin, mumax, datapoints)
if log is True:
    rlist_temp = numpy.linspace(numpy.log10(rmin), numpy.log10(rmax), datapoints)
    rlist = numpy.power(10, rlist_temp)
    mulist_temp = numpy.linspace(numpy.log10(mumin), numpy.log10(mumax), datapoints)
    mulist = numpy.power(10, mulist_temp)


if mufliped == False:
    r = rlist[x1 // datapoints]
    mu = mulist[x1 % datapoints]
else:
    mulist = numpy.flip(mulist)
    r = rlist[x1 % datapoints]
    mu = mulist[x1 // datapoints]


print("r=", r)
print("mu=", mu)


tm_array = numpy.zeros((1, 10))
tm_array[0, 0] = r
tm_array[0, 1] = mu

start = timeit.default_timer()
temp_values = selrec_exploring_r_mu_cluster(r, mu, N, l, p, k, avg, landscape, two_point_distances)

tm_array[0, 2] = temp_values[0]
tm_array[0, 3] = temp_values[1]
tm_array[0, 4] = temp_values[2]
tm_array[0, 5] = temp_values[3]
tm_array[0, 6] = temp_values[4]
tm_array[0, 7] = temp_values[5]
tm_array[0, 8] = temp_values[6]
tm_array[0, 9] = temp_values[7]


numpy.save('Time_for_exploring_perc/L{}_N{}_p{}_avg{}_dp{}_i{}_{}'.format(l, N, p, avg, datapoints, int(x1), landscape), tm_array)

stop = timeit.default_timer()
print("Time", stop - start)


f = open("timeit/T_L{}_N{}_p{}_avg{}_dp{}_{}.txt".format(l, N, p, avg, datapoints, landscape), "a")
f.write("i{}: t={}s\n".format(x1, int(stop - start)))
f.close()


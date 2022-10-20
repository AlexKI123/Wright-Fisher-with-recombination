import matplotlib.pyplot as plt
import numpy
import timeit
import numba as nb
import random
import math
from numba import prange

#################################################################################
#################################################################################
# Code to generate data and plot Fig. 12
#################################################################################
#################################################################################


@nb.njit(parallel=True)
def selrec_exploring(N, l, mu, datapoints, avg, rlist, landscape):

    t_array = numpy.zeros((datapoints, 3))
    print("rlist", rlist)

    for x1 in prange(datapoints):
        print("x", x1)
        t_array[x1, 0] = rlist[x1]
        wp = [0] * N
        wp_temp2 = [0] * N
        NL = N * l  # mutation v2 & v3
        NL_range_tempalte = list(range(NL))  # mutation v3
        selected = {0, 1, 2}  # mutation v3
        viables = set()
        lethals = set()
        neighbors = [0]  # test

        for x11 in range(avg):
            viables.clear()
            lethals.clear()
            viable_first = 0


            ###############################
            ###percolation
            if landscape == "perc":
                for x in range(2 ** l):
                    if random.random() <= p:
                        viables.add(x)
                        viable_first = x
                    else:
                        lethals.add(x)

            ###############################
            ###one_component
            if landscape == "gc":
                viables.add(0)
                ns = int(2**l*p)
                while len(viables) < ns:
                    rand_idx = random.randrange(0, len(viables))
                    choosen_viable = list(viables)[rand_idx]

                    #del neighbors[:]
                    neighbors = [0]
                    for loci in range(l):
                        neighbors.append(choosen_viable ^ (1 << loci))
                    rand_idx = random.randrange(0, len(neighbors))
                    viable_candidate = neighbors[rand_idx]
                    viables.add(viable_candidate)

                for x in range(2**l):
                    if x in viables:
                        pass
                    else:
                        lethals.add(x)
            ###############################
            rand_idx = random.randrange(0, len(viables))
            viable_first = list(viables)[rand_idx]

            viables_copy = viables.copy()
            for x3 in range(N):
                wp[x3] = viable_first
                wp_temp2[x3] = viable_first

            gen_counter = 0
            mut_counter = 0

            missing_genotypes = 1
            while missing_genotypes > 0:
                gen_counter += 1

                segMut_temp = len(set(wp))
                # if not [wp[0]] * len(wp) == wp:
                if segMut_temp > 1:
                    # selection + recombination
                    for x3 in range(N):
                        wp_temp2[x3] = wp[x3]
                    for x3 in range(N):
                        if random.random() <= rlist[x1]:
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


                # mutation v3
                flips = numpy.random.binomial(NL, mu)
                mut_counter += flips
                if flips > 0:
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
                        wp[pos // l] = wp[pos // l] ^ (1 << (pos % l))

                if flips > 0 or segMut_temp > 1:
                    for x3 in range(N):
                        if wp[x3] in viables_copy:
                            viables_copy.remove(wp[x3])
                    missing_genotypes = len(viables_copy)

            t_array[x1, 1] += 1.0*gen_counter/avg
            t_array[x1, 2] += 1.0*mut_counter/avg

    return t_array


rmin = 0.001
rmax = 1.0
datapoints = 30  #100
avg = 10000
log = True
mu = 0.01
l = 10
N = 100
p = 0.9
save = True
load = False
multipleload = True
landscape = "perc"


if log is False:
    rlist = numpy.linspace(rmin, rmax, datapoints)
if log is True:
    rlist_temp = numpy.linspace(numpy.log10(rmin), numpy.log10(rmax), datapoints)
    rlist = numpy.power(10, rlist_temp)


print("rlist", rlist)

if multipleload == False:
    if load == True:
        save = False

    print("N=", N)
    print("mu=", mu)
    print("l=", l)
    print("p=", p)
    print("datapoints:", datapoints)
    print("avg:", avg)
    print("N*L*mu", N*l*mu)

    start = timeit.default_timer()
    if load is False:
        tm_array = selrec_exploring(N, l, mu, datapoints, avg, rlist, landscape)
    else:
        tm_array = numpy.load('Time_for_exploring/TM_r_rmin{}_rmax{}_mu{}_dp{}_avg{}_l{}_N{}_p{}_{}.npy'.format(rmin, rmax, mu, datapoints, avg, l, N, p, landscape))

    stop = timeit.default_timer()
    print("Time", stop-start)

    #print(tm_array)
    if save is True:
        numpy.save('Time_for_exploring/TM_r_rmin{}_rmax{}_mu{}_dp{}_avg{}_l{}_N{}_p{}_{}.npy'.format(rmin, rmax, mu, datapoints, avg, l, N, p, landscape), tm_array)

    print(tm_array)
    plt.figure(1)
    if log is True:
        plt.xscale("log")
    plt.plot(tm_array[:, 0], tm_array[:, 1], '.-')

    plt.ylabel("Generations")
    plt.xlabel("Recombination rate $r$")
    #plt.savefig('saved_data/mr_r.pdf', bbox_inches='tight')
    plt.title("$L$={}, $N$={}, $\mu$={}, $p$={}".format(l, N, mu, p))


    plt.figure(2)
    if log is True:
        plt.xscale("log")
    plt.plot(tm_array[:, 0], tm_array[:, 2], '.-')

    plt.ylabel("Mutations")
    plt.xlabel("Recombination rate $r$")


    plt.title("$L$={}, $N$={}, $\mu$={}, $p$={}".format(l, N, mu, p))

else:
    multiplot = 2
    if multiplot == 0:
        landscape = "perc"
        l=10
        mu=0.01
        avg = 10000 # 10000 100000
        datapoints = 30  #30 100
        p1 = 1.0
        p2 = 0.9
        p3 = 0.8  # 0.88
        #p4 = 0.7
        #p5 = 0.5
        print('Time_for_exploring/TM_r_rmin{}_rmax{}_mu{}_dp{}_avg{}_l{}_N{}_p{}_{}.npy'.format(rmin, rmax, mu, datapoints, avg, l, N, p1, "gc"))
        print('Time_for_exploring/TM_r_rmin{}_rmax{}_mu{}_dp{}_avg{}_l{}_N{}_p{}_{}.npy'.format(rmin, rmax, mu, datapoints, avg, l, N, p2, landscape))
        print('Time_for_exploring/TM_r_rmin{}_rmax{}_mu{}_dp{}_avg{}_l{}_N{}_p{}_{}.npy'.format(rmin, rmax, mu, datapoints, avg, l, N, p3, landscape))
        tm_array_p1 = numpy.load('Time_for_exploring/TM_r_rmin{}_rmax{}_mu{}_dp{}_avg{}_l{}_N{}_p{}_{}.npy'.format(rmin, rmax, mu, datapoints, avg, l, N, p1, "gc"))
        tm_array_p2 = numpy.load('Time_for_exploring/TM_r_rmin{}_rmax{}_mu{}_dp{}_avg{}_l{}_N{}_p{}_{}.npy'.format(rmin, rmax, mu, datapoints, avg, l, N, p2, landscape))
        tm_array_p3 = numpy.load('Time_for_exploring/TM_r_rmin{}_rmax{}_mu{}_dp{}_avg{}_l{}_N{}_p{}_{}.npy'.format(rmin, rmax, mu, datapoints, avg, l, N, p3, landscape))
        #tm_array_p4 = numpy.load('Time_for_exploring/TM_r_rmin{}_rmax{}_mu{}_dp{}_avg{}_l{}_N{}_p{}_{}.npy'.format(rmin, rmax, mu, datapoints, avg, l, N, p4, landscape))
        #tm_array_p5 = numpy.load('Time_for_exploring/TM_r_rmin{}_rmax{}_mu{}_dp{}_avg{}_l{}_N{}_p{}_{}.npy'.format(rmin, rmax, mu, datapoints, avg, l, N, p5, landscape))

        #plt.figure(1, figsize=(4, 3))
        plt.figure(1, figsize=(5, 2.7))
        if log is True:
            plt.xscale("log")
            plt.yscale("log")
        plt.plot(tm_array_p1[:, 0], 1.0/tm_array_p1[:, 1], '--', label="$p$={}".format(p1))
        plt.plot(tm_array_p2[:, 0], 1.0/tm_array_p2[:, 1], '--', label="$p$={}".format(p2))
        plt.plot(tm_array_p3[:, 0], 1.0/tm_array_p3[:, 1], '--', label="$p$={}".format(p3))
        #plt.plot(tm_array_p4[:, 0], tm_array_p4[:, 1]/tm_array_p4[:, 1][0], '--', label="$p$={}".format(p4))
        #plt.plot(tm_array_p5[:, 0], tm_array_p5[:, 1]/tm_array_p5[:, 1][0], '--', label="$p$={}".format(p5))

        plt.ylabel("1/Generations")
        plt.xlabel("Recombination rate $r$")
        # plt.savefig('saved_data/mr_r.pdf', bbox_inches='tight')
        plt.title("$L$={}, $N$={}, $\mu$={}".format(l, N, mu))
        plt.legend(ncol=2, loc=3)
        plt.tight_layout()
        plt.savefig("T_r_gc_v2_new_{}.pdf".format(landscape))

    elif multiplot == 1:
        landscape = "gc"
        avg= 100000 #100000
        avg_2= 10000
        datapoints = 100 #100
        datapoints_2 = 30
        l5 = 5
        l6 = 6
        l7 = 7
        l8 = 8
        l9 = 9
        l10 = 10
        p = 0.9
        tm_array_p5 = numpy.load('Time_for_exploring/TM_r_rmin{}_rmax{}_mu{}_dp{}_avg{}_l{}_N{}_p{}_{}.npy'.format(rmin, rmax, mu, datapoints, avg, l5, N, p, landscape))
        tm_array_p6 = numpy.load('Time_for_exploring/TM_r_rmin{}_rmax{}_mu{}_dp{}_avg{}_l{}_N{}_p{}_{}.npy'.format(rmin, rmax, mu, datapoints, avg, l6, N, p, landscape))
        tm_array_p7 = numpy.load('Time_for_exploring/TM_r_rmin{}_rmax{}_mu{}_dp{}_avg{}_l{}_N{}_p{}_{}.npy'.format(rmin, rmax, mu, datapoints, avg, l7, N, p, landscape))
        tm_array_p8 = numpy.load('Time_for_exploring/TM_r_rmin{}_rmax{}_mu{}_dp{}_avg{}_l{}_N{}_p{}_{}.npy'.format(rmin, rmax, mu, datapoints, avg, l8, N, p, landscape))
        tm_array_p9 = numpy.load('Time_for_exploring/TM_r_rmin{}_rmax{}_mu{}_dp{}_avg{}_l{}_N{}_p{}_{}.npy'.format(rmin, rmax, mu, datapoints_2, avg_2, l9, N, p, landscape))
        tm_array_p10 = numpy.load('Time_for_exploring/TM_r_rmin{}_rmax{}_mu{}_dp{}_avg{}_l{}_N{}_p{}_{}.npy'.format(rmin, rmax, mu, datapoints_2, avg_2, l10, N, p, landscape))


        plt.figure(1, figsize=(4, 3))
        if log is True:
            plt.xscale("log")
            #plt.yscale("log")
            #plt.ylim(90, 1500)
        plt.plot(tm_array_p5[:, 0], 1/(tm_array_p5[:, 1]/tm_array_p5[:, 1][0]), '--', label="$L$={}".format(l5))
        #plt.plot(tm_array_p6[:, 0], tm_array_p6[:, 1]/tm_array_p6[:, 1][0], '--', label="$L$={}".format(l6))
        plt.plot(tm_array_p7[:, 0], 1/(tm_array_p7[:, 1]/tm_array_p7[:, 1][0]), '--', label="$L$={}".format(l7))
        #plt.plot(tm_array_p8[:, 0], tm_array_p8[:, 1]/tm_array_p8[:, 1][0], '--', label="$L$={}".format(l8))
        #plt.plot(tm_array_p9[:, 0], tm_array_p9[:, 1]/tm_array_p9[:, 1][0], '--', label="$L$={}".format(l9))
        plt.plot(tm_array_p10[:, 0], 1/(tm_array_p10[:, 1]/tm_array_p10[:, 1][0]), '--', label="$L$={}".format(l10))


        # plt.plot(tm_array_p3[:, 0], tm_array_p5[:, 1], '--', label="$p$={}".format(p3))

        plt.ylabel("Generations($r$)/Generations($r=0$)")
        #plt.ylabel("$t(r)$/$t(r=0)$")

        plt.xlabel("Recombination rate $r$")
        # plt.savefig('saved_data/mr_r.pdf', bbox_inches='tight')
        plt.title("$p$={}, $N$={}, $\mu$={}".format(p, N, mu))
        #plt.legend(ncol=1, bbox_to_anchor=(0.3, 0.7))
        plt.legend(ncol=3)
        #plt.legend(ncol=2, loc=2)
        plt.tight_layout()
        plt.savefig("T_r_gc.pdf")

        plt.figure(2)
        if log is True:
            plt.xscale("log")
        plt.plot(tm_array_p5[:, 0], tm_array_p5[:, 2], '--', label="$L$={}".format(l5))
        plt.plot(tm_array_p6[:, 0], tm_array_p6[:, 2], '--', label="$L$={}".format(l6))
        plt.plot(tm_array_p7[:, 0], tm_array_p7[:, 2], '--', label="$L$={}".format(l7))
        plt.plot(tm_array_p8[:, 0], tm_array_p8[:, 2], '--', label="$L$={}".format(l8))
        plt.plot(tm_array_p9[:, 0], tm_array_p9[:, 2], '--', label="$L$={}".format(l9))

        plt.ylabel("Total mutations")
        plt.xlabel("Recombination rate $r$")

        plt.title("$L$={}, $N$={}, $\mu$={}".format(l, N, mu))
        plt.legend(ncol=1)
    else:
        landscape = "perc"
        avg = 100000  # 100000
        avg_2 = 10000
        datapoints = 100  # 100
        datapoints_2 = 30
        l5 = 5
        mu5 = 0.02
        l7 = 7
        mu7 = 0.01428571428
        l10 = 10
        mu10 = 0.01
        p = 0.9

        print('Time_for_exploring/TM_r_rmin{}_rmax{}_mu{}_dp{}_avg{}_l{}_N{}_p{}_{}.npy'.format(rmin, rmax, mu10,
                                                                                              datapoints_2, avg_2, l10,
                                                                                              N, p, landscape))
        print('Time_for_exploring/TM_r_rmin{}_rmax{}_mu{}_dp{}_avg{}_l{}_N{}_p{}_{}.npy'.format(rmin, rmax, mu7,
                                                                                              datapoints_2, avg, l7, N, p,
                                                                                              landscape))
        print('Time_for_exploring/TM_r_rmin{}_rmax{}_mu{}_dp{}_avg{}_l{}_N{}_p{}_{}.npy'.format(rmin, rmax, mu5,
                                                                                              datapoints_2, avg, l5, N, p,
                                                                                              landscape))

        tm_array_p10 = numpy.load(
            'Time_for_exploring/TM_r_rmin{}_rmax{}_mu{}_dp{}_avg{}_l{}_N{}_p{}_{}.npy'.format(rmin, rmax, mu10,
                                                                                              datapoints_2, avg_2, l10,
                                                                                              N, p, landscape))

        tm_array_p7 = numpy.load(
            'Time_for_exploring/TM_r_rmin{}_rmax{}_mu{}_dp{}_avg{}_l{}_N{}_p{}_{}.npy'.format(rmin, rmax, mu7,
                                                                                              datapoints_2, avg, l7, N, p,
                                                                                              landscape))

        tm_array_p5 = numpy.load(
            'Time_for_exploring/TM_r_rmin{}_rmax{}_mu{}_dp{}_avg{}_l{}_N{}_p{}_{}.npy'.format(rmin, rmax, mu5,
                                                                                              datapoints_2, avg, l5, N, p,
                                                                                              landscape))

        plt.figure(1, figsize=(5, 2.7))
        if log is True:
            plt.xscale("log")
            # plt.yscale("log")
            # plt.ylim(90, 1500)
        plt.plot(tm_array_p5[:, 0], 1 / (tm_array_p5[:, 1] / tm_array_p5[:, 1][0]), '--', label="$L$={}".format(l5), color="C3")
        # plt.plot(tm_array_p6[:, 0], tm_array_p6[:, 1]/tm_array_p6[:, 1][0], '--', label="$L$={}".format(l6))
        plt.plot(tm_array_p7[:, 0], 1 / (tm_array_p7[:, 1] / tm_array_p7[:, 1][0]), '--', label="$L$={}".format(l7), color="C4")
        #plt.plot(tm_array_p8[:, 0], 1 / (tm_array_p8[:, 1] / tm_array_p8[:, 1][0]), '--', label="$L$={}".format(l8))
        # plt.plot(tm_array_p9[:, 0], tm_array_p9[:, 1]/tm_array_p9[:, 1][0], '--', label="$L$={}".format(l9))
        plt.plot(tm_array_p10[:, 0], 1 / (tm_array_p10[:, 1] / tm_array_p10[:, 1][0]), '--', label="$L$={}".format(l10), color="C1")

        # plt.plot(tm_array_p3[:, 0], tm_array_p5[:, 1], '--', label="$p$={}".format(p3))

        #plt.ylabel("Generations($r$)/Generations($r=0$)")
        plt.ylabel("Generations($r=0$)/Generations($r$)")

        # plt.ylabel("$t(r)$/$t(r=0)$")

        plt.xlabel("Recombination rate $r$")
        # plt.savefig('saved_data/mr_r.pdf', bbox_inches='tight')
        plt.title("$p$={}, $N$={}, $L\mu$={}".format(p, N, 0.1))
        # plt.legend(ncol=1, bbox_to_anchor=(0.3, 0.7))
        plt.legend(ncol=3)
        # plt.legend(ncol=2, loc=2)
        plt.tight_layout()
        plt.savefig("T_r_gc_l_relative_{}.pdf".format(landscape))


plt.show()

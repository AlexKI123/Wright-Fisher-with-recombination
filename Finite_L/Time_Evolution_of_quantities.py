import matplotlib.pyplot as plt
import numpy
import timeit
import numba as nb
import random
import math
from numba import prange
from modules import ham_dist_fct_nb
from modules import mutational_robustness_nb_smr2_hd
import scipy.special
#import seaborn as sns


#################################################################################
#################################################################################
# Code to generate data and plot Fig. 14, S12, S13, S14, S15. (Generates large data files).
#################################################################################
#################################################################################

def tsplot(data, **kw):
    x = numpy.arange(data.shape[1])
    x = numpy.arange(1, len(x)+1)
    est = numpy.mean(data, axis=0)
    sd = numpy.std(data, axis=0)
    cis = (est - sd, est + sd)
    plt.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
    plt.plot(x, est, **kw)
    plt.margins(x=0)


def tsplot_r0(data, **kw):
    x = numpy.arange(data.shape[1])
    x = numpy.arange(1, len(x)+1)
    est = numpy.mean(data, axis=0)
    sd = numpy.std(data, axis=0)
    cis = (est - sd, est + sd)
    plt.fill_between(x, cis[0], cis[1], alpha=0.2, color="C0")
    plt.plot(x, est, color="C0", label="$\overline{m}(r=0)$")
    plt.margins(x=0)


def tsplot_r1(data, **kw):
    x = numpy.arange(data.shape[1])
    x = numpy.arange(1, len(x)+1)
    est = numpy.mean(data, axis=0)
    sd = numpy.std(data, axis=0)
    cis = (est - sd, est + sd)
    plt.fill_between(x, cis[0], cis[1], alpha=0.2, color="C1")
    plt.plot(x, est, color="C1", label="$\overline{m}(r=1)$")
    plt.margins(x=0)


@nb.njit
def pfunc(N, l, r, mu, landscape, generations, two_point_distances):
    wp = [0] * N
    wp_temp2 = [0] * N
    NL = N * l  # mutation v2 & v3
    NL_range_tempalte = list(range(NL))  # mutation v3
    selected = {0, 1, 2}  # mutation v3
    viables = {1, 2, 3} # set() helps numba to  infer the type
    lethals = {1, 2, 3} # set() helps numba to  infer the type
    neighbors = [0]  # test

    viables.clear()
    lethals.clear()


    if landscape == "perc":
        ns = int(2 ** l * p)
        viables = set(numpy.random.choice(2 ** l, ns, replace=False))
        for x in range(2 ** l):
            if x in viables:
                pass
            else:
                lethals.add(x)

        rand_idx = random.randrange(0, len(viables))
        viable_first = list(viables)[rand_idx]


    ###############################
    ###one_component
    if landscape == "gc":
        viables.add(0)
        ns = int(2 ** l * p)
        while len(viables) < ns:
            rand_idx = random.randrange(0, len(viables))
            choosen_viable = list(viables)[rand_idx]

            #del neighbors[:]
            neighbors = []
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

        rand_idx = random.randrange(0, len(viables))
        viable_first = list(viables)[rand_idx]
    ###############################



    viables_copy = viables.copy()
    for x3 in range(N):
        wp[x3] = viable_first
        wp_temp2[x3] = viable_first

    mut_counter = 0
    mut_counter_temp = 0
    #print("len(viables)", len(viables))
    #print("percentage_list", percentage_list)

    m_value = [0.0]*generations
    avg_hd = [0.0]*generations
    nogenotypes = [0]*generations
    nosegmut = [0]*generations
    explored = [0.0]*generations

    missing_genotypes = len(viables_copy) - 1
    for gen in range(generations):

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
                    #mut_counter += 1
                    mut_counter_temp += 1
                    wp[mutation_in_individual] = wp[mutation_in_individual] ^ (1 << mutation_at_loci)
            if mut_counter_temp > 0:
                mut_counter += 1

        """
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
        
        """
        if mut_counter_temp > 0 or segMut_temp > 1 or gen == 0:
        #if flips > 0 or segMut_temp > 1 or gen == 0:
            for x3 in range(N):
                if wp[x3] in viables_copy:
                    viables_copy.remove(wp[x3])
            missing_genotypes = len(viables_copy)

            temp_values = mutational_robustness_nb_smr2_hd(N, wp, viables, l, two_point_distances)

        m_value[gen] = 1.0 * temp_values[0]
        avg_hd[gen] = 1.0 * temp_values[1]
        nogenotypes[gen] = 1.0 * temp_values[2]
        nosegmut[gen] = 1.0 * temp_values[3]
        explored[gen] = 1.0 * (len(viables)-missing_genotypes)/len(viables)

    return m_value, avg_hd, nogenotypes, nosegmut, explored


@nb.njit(parallel=True)
def selrec_time_evolution(N, l, r, mu, avg, landscape, generations, two_point_distances):

    #percentage_list_all = numpy.zeros(44)#numpy.zeros(int(2**l*p))
    #percentage_list_all = numpy.zeros(int(2**l*p))
    m_value = numpy.zeros((avg, generations))
    avg_hd = numpy.zeros((avg, generations))
    nosegmut = numpy.zeros((avg, generations))
    nogenotypes = numpy.zeros((avg, generations))
    explored = numpy.zeros((avg, generations))

    for i in prange(avg):
        m_value_1, avg_hd_1, nogenotypes_1, nosegmut_1, explored_1 = pfunc(N, l, r, mu, landscape, generations, two_point_distances)
        m_value[i] = m_value_1
        avg_hd[i] = avg_hd_1
        nogenotypes[i] = nogenotypes_1
        nosegmut[i] = nosegmut_1
        explored[i] = explored_1

    return m_value, avg_hd, nogenotypes, nosegmut, explored


avg = 10000
r = 1.0
mu = 0.0001
l = 10
N = 100
p = 0.5
generations = 50000
save = False
load = True
multipleload = True  # show several quantities?
fourfourGrid = True  # show several figures?
landscape = "perc"
plotall = True
measure = 0
#0 m
#1 avg_hd
#2 NoGen
#3 NoSegMut
#4 Explored

print("N*l*mu", N * l * mu)
two_point_distances = scipy.special.binom(N, 2)


if multipleload == False:
    if load == True:
        save = False

    print("N=", N)
    print("r=", r)
    print("mu=", mu)
    print("l=", l)
    print("p=", p)
    print("avg:", avg)

    start = timeit.default_timer()

    if load is False:
        temp_array = selrec_time_evolution(N, l, r, mu, avg, landscape, generations, two_point_distances)
    else:
        temp_array = numpy.load('Time_for_exploring/temp_array_mu{}_r{}_avg{}_l{}_N{}_p{}_gen{}_{}.npy'.format(mu, r, avg, l, N, p, generations, landscape))



    #print(percentage_list_all)
    stop = timeit.default_timer()
    print("Time", stop-start)

    #print(tm_array)
    if save is True:
        numpy.save('Time_for_exploring/temp_array_mu{}_r{}_avg{}_l{}_N{}_p{}_gen{}_{}.npy'.format(mu, r, avg, l, N, p, generations, landscape), temp_array)

    if plotall == False:
        plt.figure(1)

        plt.xlabel("Generations")
        plt.title("$L$={}, $N$={}, $r$={}, $\mu={}$ $p$={}".format(l, N, r, mu, p))
        temp_array = numpy.array(temp_array)
        for ii in range(3):
            plt.plot(numpy.arange(1, len(temp_array[measure, ii])+1), temp_array[measure,ii], alpha=0.5)

        #plt.plot(m_value.mean(0))
        #plt.plot(numpy.average(m_value, axis=0))
        #plt.plot(explored, label="explored")
        #plt.plot(avg_hd, label="avg_hd")
        #plt.plot(nogenotypes, label="nogen")
        #plt.plot(nosegmut, label="segmut")
        #plt.legend()

        #sns.tsplot(data=m_value)
        #x = numpy.tile(numpy.arange(generations), avg)
        #y = m_value.flatten()
        #ax = sns.lineplot(x, y, estimator="rolling_mean", ci=95)

        tsplot(temp_array[measure], color="C0")
        #tsplot(explored)

    if plotall == True:
        from mpl_toolkits.axes_grid1 import host_subplot
        import mpl_toolkits.axisartist as AA

        host = host_subplot(111, axes_class=AA.Axes)
        plt.subplots_adjust(right=0.75)

        par1 = host.twinx()
        par2 = host.twinx()

        offset = 60
        new_fixed_axis = par2.get_grid_helper().new_fixed_axis
        par2.axis["right"] = new_fixed_axis(loc="right",
                                            axes=par2,
                                            offset=(offset, 0))

        par2.axis["right"].toggle(all=True)

        #host.set_xlim(0, 2)
        host.set_ylim(0, 1)

        host.set_xlabel("Generations")
        host.set_ylabel("$m$ / explored ")
        par1.set_ylabel("Avg_hd / Seg_mut")
        par2.set_ylabel("NoGen")

        p1, = host.plot(explored, label="explored")
        p2, = host.plot(m_value, label="$m$")
        p3, = par1.plot(avg_hd, label="avg_hd")
        p4, = par1.plot(nosegmut, label="nosegmut")
        p5, = par2.plot(nogenotypes, label="nogenotypes")

        par1.set_ylim(0, l)
        par2.set_ylim(0, N)

        host.legend()

        host.axis["left"].label.set_color(p1.get_color())
        par1.axis["right"].label.set_color(p2.get_color())
        par2.axis["right"].label.set_color(p3.get_color())

        plt.draw()

else:
    if fourfourGrid == False:
        r0 = 0.0
        r1 = 1.0

        temp_array_r0 = numpy.load('Time_for_exploring/temp_array_mu{}_r{}_avg{}_l{}_N{}_p{}_gen{}_{}.npy'.format(mu, r0, avg, l, N, p, generations, landscape))
        temp_array_r1 = numpy.load('Time_for_exploring/temp_array_mu{}_r{}_avg{}_l{}_N{}_p{}_gen{}_{}.npy'.format(mu, r1, avg, l, N, p, generations, landscape))


        plt.figure(1, figsize=(4, 3))

        #plt.plot(m_value_r0.mean(0), label="$m(r=0)$")
        #plt.plot(m_value_r1.mean(0), label="$m(r=1)$")



        #for ii in range(10):
            # print(m_value[ii])
        #    plt.plot(m_value_r0[ii], color="C0", alpha=0.2)
        #    plt.plot(m_value_r1[ii], color="C1", alpha=0.2)
            # plt.plot(explored[ii], alpha=0.1)

        #plt.plot(m_value_r0.mean(0), color="C0")
        #plt.plot(m_value_r1.mean(0), color="C1")

        tsplot_r0(temp_array_r0[measure])
        tsplot_r1(temp_array_r1[measure])
        #plt.ylim(0.3,0.95)
        plt.xlim((1, generations))
        if mu == 0.1:
            plt.xticks([1,10,20,30,40,50])
        if mu == 0.01:
            plt.xticks([1,100,200,300,400,500])
        if mu == 0.001:
            plt.xticks([1,1000,2000,3000,4000,5000])
        if mu == 0.00001:
            plt.xticks([1,10000,20000,30000,40000,50000])

        #plt.legend(ncol=2)
        plt.legend()


        plt.xlabel("Generations")
        plt.ylabel("Mutational robustness $m$")
        plt.title("$N$={}, $L$={}, $\mu={}$, $p$={}".format(N, l, mu, p))
        #plt.legend()
        plt.tight_layout()

    else:
        r0 = 0.0
        r1 = 1.0
        mu_list = [0.1,0.01,0.001,0.0001]
        generations_list = [50,500,5000,50000]
        avg_list = [10000, 10000, 10000, 5000]
        avg_list2 = [10000, 10000, 10000, 10000]

        fig, axs = plt.subplots(2, 2, figsize=(6, 6))

        for x in range(len(mu_list)):
            #m_value_r0 = numpy.load('Time_for_exploring/m_value_mu{}_r{}_avg{}_l{}_N{}_p{}_gen{}_{}.npy'.format(mu_list[x], r0, avg_list[x], l, N, p, generations_list[x], landscape))
            #m_value_r1 = numpy.load('Time_for_exploring/m_value_mu{}_r{}_avg{}_l{}_N{}_p{}_gen{}_{}.npy'.format(mu_list[x], r1, avg_list[x], l, N, p, generations_list[x], landscape))

            temp_array_r0 = numpy.load('Time_for_exploring/temp_array_mu{}_r{}_avg{}_l{}_N{}_p{}_gen{}_{}.npy'.format(mu_list[x], r0, avg_list[x], l, N, p, generations_list[x], landscape))[measure]
            temp_array_r1 = numpy.load('Time_for_exploring/temp_array_mu{}_r{}_avg{}_l{}_N{}_p{}_gen{}_{}.npy'.format(mu_list[x], r1, avg_list2[x], l, N, p, generations_list[x], landscape))[measure]

            aa = x//2
            bb = x%2
            x_value = numpy.arange(1, generations_list[x] + 1)
            label_list_r0 = ["$m(r=0)$", "$d_{pw}(r=0)$", "$Y(r=0)$", "$S(r=0)$", "Explored($r=0$)"]
            label_list_r1 = ["$m(r=1)$", "$d_{pw}(r=1)$", "$Y(r=1)$", "$S(r=1)$", "Explored($r=1$)"]

            est = numpy.mean(temp_array_r0, axis=0)
            sd = numpy.std(temp_array_r0, axis=0)
            cis = (est - sd, est + sd)
            axs[aa, bb].fill_between(x_value, cis[0], cis[1], alpha=0.2, color="C0")
            if aa == 1 and bb == 1:
                l1 = axs[aa, bb].plot(x_value, est, color="C0", label=label_list_r0[measure])
            else:
                axs[aa, bb].plot(x_value, est, color="C0", label=label_list_r0[measure])
            axs[aa, bb].margins(x=0)

            est = numpy.mean(temp_array_r1, axis=0)
            sd = numpy.std(temp_array_r1, axis=0)
            cis = (est - sd, est + sd)
            axs[aa, bb].fill_between(x_value, cis[0], cis[1], alpha=0.2, color="C1")
            if aa == 1 and bb == 1:
                l2 = axs[aa, bb].plot(x_value, est, color="C1", label=label_list_r1[measure])
            else:
                axs[aa, bb].plot(x_value, est, color="C1", label=label_list_r1[measure])
            axs[aa, bb].margins(x=0)

            if measure == 0:
                #axs[aa, bb].set_ylim(0.3, 1.0)
                axs[aa, bb].set_ylim(0.2, 0.8)
                #axs[aa, bb].text(generations_list[x]/8, 0.91, '$\mu={}$'.format(mu_list[x]), verticalalignment='bottom', horizontalalignment='left', fontsize=15)
                axs[aa, bb].text(0.1, 0.93, '$\mu={}$'.format(mu_list[x]), horizontalalignment='left', verticalalignment='center', transform=axs[aa, bb].transAxes, fontsize=15)
            if measure == 1:
                #axs[aa, bb].set_ylim(0.0, 5.0)
                #axs[aa, bb].text(generations_list[x]/2, 1.7, '$\mu={}$'.format(mu_list[x]), verticalalignment='bottom', horizontalalignment='left', fontsize=15)
                axs[aa, bb].text(0.3, 0.4, '$\mu={}$'.format(mu_list[x]), horizontalalignment='left', verticalalignment='center', transform=axs[aa, bb].transAxes, fontsize=15)
            if measure == 2:
                #axs[aa, bb].set_ylim(0.0, 100.0)
                #axs[aa, bb].text(generations_list[x]/2, 50, '$\mu={}$'.format(mu_list[x]), verticalalignment='bottom', horizontalalignment='left', fontsize=15)
                #axs[aa, bb].text(0.15, 0.7, '$\mu={}$'.format(mu_list[x]), horizontalalignment='left', verticalalignment='center', transform=axs[aa, bb].transAxes, fontsize=15)
                axs[aa, bb].text(0.3, 0.2, '$\mu={}$'.format(mu_list[x]), horizontalalignment='left', verticalalignment='center', transform=axs[aa, bb].transAxes, fontsize=15)
            if measure == 3:
                #axs[aa, bb].set_ylim(0.0, 10.1)
                #xs[aa, bb].text(generations_list[x]/2, 5, '$\mu={}$'.format(mu_list[x]), verticalalignment='bottom', horizontalalignment='left', fontsize=15)
                axs[aa, bb].text(0.15, 0.75, '$\mu={}$'.format(mu_list[x]), horizontalalignment='left', verticalalignment='center', transform=axs[aa, bb].transAxes, fontsize=15)
            if measure == 4:
                #axs[aa, bb].set_ylim(0.0, 1.0)
                #axs[aa, bb].text(generations_list[x]/10, 0.8, '$\mu={}$'.format(mu_list[x]), verticalalignment='bottom', horizontalalignment='left', fontsize=15)
                axs[aa, bb].text(0.1, 0.9, '$\mu={}$'.format(mu_list[x]), horizontalalignment='left', verticalalignment='center', transform=axs[aa, bb].transAxes, fontsize=15)


            axs[aa, bb].set_xlim(1, generations_list[x])
            #axs[aa, bb].set_title('$\mu={}$'.format(mu_list[x]))


            if mu_list[x] == 0.1:
                axs[aa, bb].set_xticks([1, 25, 50])
            if mu_list[x] == 0.01:
                axs[aa, bb].set_xticks([1, 250, 500])
            if mu_list[x] == 0.001:
                axs[aa, bb].set_xticks([1, 2500, 5000])
            if mu_list[x] == 0.0001:
                axs[aa, bb].set_xticks([1, 25000, 50000])

            if aa == 1:
                axs[aa, bb].set_xlabel("Generations")

        #plt.legend(loc="center right", bbox_to_anchor=[0, 1], ncol=2, shadow=True, title="Legend", fancybox=True)
        #plt.legend(loc="center right", borderaxespad=0.1)

        #fig.suptitle('Main title')  # or plt.suptitle('Main title')
        fig.suptitle("$N$={}, $L$={}, $p$={}".format(N, l, p), fontsize=12)


        plt.tight_layout()

        # Create the legend
        fig.legend([l1, l2],  # The line objects
                   labels=[label_list_r0[measure], label_list_r1[measure]],  # The labels for each line
                   #loc="upper center",  # Position of legend
                   loc="lower center",
                   borderaxespad=0.1,  # Small spacing around legend box
                   ncol=2,
                   fontsize=11
                   )

        # Adjust the scaling factor to fit your legend text completely outside the plot
        # (smaller value results in more space being made for the legend)

        #plt.subplots_adjust(top=0.92)
        plt.subplots_adjust(bottom=0.13)

plt.show()

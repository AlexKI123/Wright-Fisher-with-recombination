import numpy as np
import numpy
import numba as nb
import random
import timeit
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
import scipy.special
from modules import selrec_vl
#from modules import mutation_vl
from modules import clean_fixed_mutations_m7, mutation_vl_poisson


#################################################################################
#################################################################################
# Code to generate data for Fig. 7.
#################################################################################
#################################################################################


def selrec_mut():
    mutation_counter = 0
    d1_distribution = np.zeros((generations, N))
    wp = [set() for _ in range(N)]
    fixed_mutations = set()
    explored_genotypes = {()}
    explored_viable_genotypes = {()}
    explored_genotypes_counter = 0
    explored_viable_genotypes_counter = 0
    viables = {()}
    lethals = set()

    start1 = timeit.default_timer()

    for gen in range(generations):
        if gen % 10000 == 0:
            print("gen", gen)
            stop1 = timeit.default_timer()
            print("Elapsed time: ", stop1-start1)
            print("Estimated time: ", (stop1-start1)/(gen+1)*generations)

        # Evolution
        wp, viables, lethals = selrec_vl(wp, N, r, p, viables, lethals)
        wp, mutation_counter, viables, lethals = mutation_vl_poisson(wp, N, U, p, mutation_counter, viables, lethals)

        # Clean fixed mutation
        wp, fixed_mutations, viables, lethals, explored_genotypes, explored_viable_genotypes, explored_genotypes_counter_temp, explored_viable_genotypes_counter_temp = clean_fixed_mutations_m7(wp, N, fixed_mutations, viables, lethals, explored_genotypes, explored_viable_genotypes)
        explored_genotypes_counter += explored_genotypes_counter_temp
        explored_viable_genotypes_counter += explored_viable_genotypes_counter_temp

        # Analysis
        for x1 in range(N):
            d1_distribution[gen, x1] = len(wp[x1])

        sum([len(wp[x1]) for x1 in range(N)])/N
        #number_segregating_mutations[gen] = len(set.union(*wp))


    return d1_distribution


N = 100
U = 0.1  #0.002  #0.00004 #0.1  # mutation rate
r = 0.01  #0.326
generations = 1_000_000  #1000000   3000000
p = 1.0
discarded_generations = int(generations/10)
analytical = False
load = False
save = False
both = False
if load is False:
    both = False

print("N=", N)
print("U=", U)
print("N*U =", N*U)
print("p=",p)
print("r =", r)
print("Generations=", generations)
print("discarded generations", discarded_generations)
print("Save=", save)


if load is False:
    startfull = timeit.default_timer()
    d1_distribution = selrec_mut()

    stopfull = timeit.default_timer()
    print("time", stopfull-startfull)

    print("max distance=", int(np.max(d1_distribution)))
    d1_distribution_list = np.zeros(int(np.max(d1_distribution)+1))
    for t in range(discarded_generations, generations):
        for i in range(N):
            d1_distribution_list[int(d1_distribution[t, i])] += 1
    d1_distribution_list = d1_distribution_list/((generations-discarded_generations)*N)
    print(sum(d1_distribution_list))

    average_d = 0
    for d in range(int(np.max(d1_distribution)+1)):
        average_d += d*d1_distribution_list[d]
    print("average_d", average_d)

    var_d = 0
    for d in range(int(np.max(d1_distribution)+1)):
        var_d += d1_distribution_list[d]*(d-average_d)**2
    print("var_d", var_d)
    std_d = var_d**(1/2)
    print("std_d", std_d)

    d1_distribution_list_max = np.max(np.where(d1_distribution_list))
    d1_distribution_list_min = np.min(np.where(d1_distribution_list))

    if save is True:
        np.save('saved_data/Cloud_CrossSection_N{}_U{}_r{}_p{}_gen{}_m2.npy'.format(N, U, r, p, generations), d1_distribution_list)

else:
    if both is False:
        d1_distribution_list = np.load('saved_data/Cloud_CrossSection_N{}_U{}_r{}_p{}_gen{}_m2.npy'.format(N, U, r, p, generations))
        d1_distribution_list_max = np.max(np.where(d1_distribution_list))
        d1_distribution_list_min = np.min(np.where(d1_distribution_list))

        average_d = 0
        for d in range(len(d1_distribution_list)):
            average_d += d * d1_distribution_list[d]
        print("average_d", average_d)

        var_d = 0
        for d in range(len(d1_distribution_list)):
            var_d += d1_distribution_list[d] * (d - average_d) ** 2
        print("var_d", var_d)
        std_d_r0 = var_d ** (1 / 2)
        print("std_d_r0", std_d_r0)


    else:
        d1_distribution_list_r0 = np.load('saved_data/Cloud_1dshape_N{}_U{}_r{}_gen{}_m2.npy'.format(N, U, 0.0, generations))
        d1_distribution_list_r1 = np.load('saved_data/Cloud_1dshape_N{}_U{}_r{}_gen{}_m2.npy'.format(N, U, 1.0, generations))
        d1_distribution_list_max_r0 = np.max(np.where(d1_distribution_list_r0))
        d1_distribution_list_max_r1 = np.max(np.where(d1_distribution_list_r1))
        d1_distribution_list_min_r0 = np.min(np.where(d1_distribution_list_r0))
        d1_distribution_list_min_r1 = np.min(np.where(d1_distribution_list_r1))
        average_d_r0 = 0
        for d in range(len(d1_distribution_list_r0)):
            average_d_r0 += d * d1_distribution_list_r0[d]
        print("average_d_r0", average_d_r0)

        average_d_r1 = 0
        for d in range(len(d1_distribution_list_r1)):
            average_d_r1 += d * d1_distribution_list_r1[d]
        print("average_d_r1", average_d_r1)

        var_d_r0 = 0
        for d in range(len(d1_distribution_list_r0)):
            var_d_r0 += d1_distribution_list_r0[d] * (d - average_d_r0) ** 2
        print("var_d_r0", var_d_r0)
        std_d_r0 = var_d_r0 ** (1 / 2)
        print("std_d_r0", std_d_r0)

        var_d_r1 = 0
        for d in range(len(d1_distribution_list_r1)):
            var_d_r1 += d1_distribution_list_r1[d] * (d - average_d_r1) ** 2
        print("var_d_r1", var_d_r1)
        std_d_r1 = var_d_r1 ** (1 / 2)
        print("std_d_r1", std_d_r1)


mu = N*U
variance = mu
sigma = math.sqrt(variance)
#x = np.linspace(mu - 5*sigma, mu + 5*sigma, 200)
if both is False:
    x_r0 = np.linspace(d1_distribution_list_min, d1_distribution_list_max, 200)
    x_r1 = np.linspace(d1_distribution_list_min, d1_distribution_list_max, 200)
else:
    x_r0 = np.linspace(0.14+d1_distribution_list_min_r0, d1_distribution_list_max_r0, 200)
    x_r1 = np.linspace(d1_distribution_list_min_r1, d1_distribution_list_max_r1, 200)

mu = N*U*p
poisson_y = np.exp(-mu)*np.power(mu, x_r1)/scipy.special.factorial(x_r1)
poisson_y2 = np.exp(-2*mu)*np.power(2*mu, x_r1)/scipy.special.factorial(x_r1)

poisson_y3 = np.exp(-average_d)*np.power(average_d, x_r1)/scipy.special.factorial(x_r1)


gamma_y = np.zeros(len(x_r0))
ii = 0
N2 = N
if analytical is True:
    for t in x_r0:
        print("t", t)
        t = t/(N*U*p)
        for i in range(2, N2):
            temp_jj = 1.0
            for j in [jt for jt in range(2, N2) if jt != i]:
                temp_jj *= scipy.special.binom(j, 2)/(scipy.special.binom(j, 2)-scipy.special.binom(i, 2))
            gamma_y[ii] += 1.0/(N*U*p) * scipy.special.binom(i, 2) * math.exp(-scipy.special.binom(i, 2) * t)*temp_jj

            #gamma_y[i] += U*scipy.special.binom(s, 2)*math.exp(-scipy.special.binom(s, 2)*xi)*(2.0*s-1.0)*(-1.0)**s*scipy.special.factorial(1.0*N)*scipy.special.factorial(1.0*N-1.0)/(scipy.special.factorial(1.0*N-s)*scipy.special.factorial(1.0*N+s-1.0))
        ii += 1


print("mu", mu)
print("sigma", sigma)

if both is True:
    newvalues_r0 = np.copy(d1_distribution_list_r0)
    newvalues_r0[newvalues_r0==0] = np.nan
    newvalues_r1 = np.copy(d1_distribution_list_r1)
    newvalues_r1[newvalues_r1==0] = np.nan

    print(d1_distribution_list_r1)
    print(newvalues_r1)

plt.figure(1, figsize=(4, 4))
if both is False:
    plt.title("$N={}$".format(N) + ", $U={}$".format(U) + ", $r={}$".format(r)+ ", $p={}$".format(p))
    plt.plot(d1_distribution_list, ".", label="Data", color="C3")
else:
    plt.title("$N={}$".format(N) + ", $U={}$".format(U) + ", $N U={}$".format(N*U))
    plt.plot(newvalues_r0, ".", label= '$r=0$: $\overline{{d}}$=${}$, Var($d$)$\\approx{}$'.format(int(average_d_r0), int(var_d_r0)), color="C0", alpha=0.5)
    plt.plot(newvalues_r1, ".", label= '$r=1$: $\overline{{d}}$=${}$, Var($d$)$\\approx{}$'.format(int(average_d_r1), int(var_d_r1)), color="C1", alpha=0.5)

plt.plot(x_r0, gamma_y, label="$h(d,r=0)$", color="C0")

plt.plot(x_r1, poisson_y2, label="Poisson dist.: $\lambda =2 N U$", color="C1")
plt.plot(x_r1, poisson_y3, label="Poisson dist.: $\lambda =2 N U$", color="C2")


plt.xlabel("Hamming distance")
plt.ylabel("Frequency")
#plt.plot(d1_distribution_list[0:np.max(np.where(d1_distribution_list))+1])
plt.legend()


plt.figure(2, figsize=(4, 4))
if both is False:
    plt.title("$N={}$".format(N) + ", $U={}$".format(U) + ", $r={}$".format(r) + ", $p={}$".format(p))
    plt.semilogy(d1_distribution_list[0:np.max(np.where(d1_distribution_list))+1], ".", label="Data", color="C3")
else:
    plt.title("$N={}$".format(N) + ", $U={}$".format(U) + ", $N U={}$".format(N*U))
#    plt.semilogy(newvalues_r0, ".", label= '$r=0$: $\overline{{d}}$=${}$, Var($d$)$\\approx{}$'.format(int(average_d_r0), int(var_d_r0)), color="C0", alpha=0.5)
#    plt.semilogy(newvalues_r1, ".", label= '$r=1$: $\overline{{d}}$=${}$, Var($d$)$\\approx{}$'.format(int(average_d_r1), int(var_d_r1)), color="C1", alpha=0.5)
    plt.semilogy(newvalues_r0, ".", label= '$r=0$: $\overline{{d}}$=${}$, Var($d$)$\\approx{}$'.format(round(average_d_r0, 2), round(var_d_r0, 2)), color="C0", alpha=0.5)
    plt.semilogy(newvalues_r1, ".", label= '$r=1$: $\overline{{d}}$=${}$, Var($d$)$\\approx{}$'.format(round(average_d_r1, 2), round(var_d_r1, 2)), color="C1", alpha=0.5)


plt.semilogy(x_r0, gamma_y, label="$h(d,r=0)$", color="C0")
plt.semilogy(x_r1, poisson_y2, label="$h(d,r=1)$", color="C1")
plt.semilogy(x_r1, poisson_y3, label="$h(d,r=1)$", color="C2")


plt.xlabel("Hamming distance $d$")
plt.ylabel("Frequency")
if both is False:
    plt.ylim(np.min(d1_distribution_list), np.max(d1_distribution_list))
else:
    plt.ylim(np.min(d1_distribution_list_r1), np.max(d1_distribution_list_r1))
plt.legend()
plt.tight_layout()
plt.show()





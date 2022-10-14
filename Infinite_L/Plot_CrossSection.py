import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.special

#################################################################################
#################################################################################
# Code to plot figure 7
#################################################################################
#################################################################################

N = 100  # population size
U = 0.1  # mutation rate
p = 0.5
generations = 1_000_000  # time steps
analytics = True
rList = [0.0, 1.0, 0.1, 0.01]

d1_distribution_list = []
for i in range(len(rList)):
    d1_distribution_list.append(np.load('saved_data/Cloud_CrossSection_N{}_U{}_r{}_p{}_gen{}_m2.npy'.format(N, U, rList[i], p, generations)))


d1_distribution_list_max_r0 = np.max(np.where(d1_distribution_list[0]))
d1_distribution_list_max_r1 = np.max(np.where(d1_distribution_list[0]))
d1_distribution_list_min_r0 = np.min(np.where(d1_distribution_list[0]))
d1_distribution_list_min_r1 = np.min(np.where(d1_distribution_list[1]))

if analytics == True:
    average_d_list =[]
    for i in range(len(rList)):
        average_d = 0
        for d in range(len(d1_distribution_list[i])):
            average_d += d * d1_distribution_list[i][d]
        average_d_list.append(average_d)

    print("average_d_list", average_d_list)


    var_d_list = []
    std_d_list = []
    for i in range(len(rList)):
        var_d = 0
        for d in range(len(d1_distribution_list[i])):
            var_d += d1_distribution_list[i][d] * (d - average_d_list[i]) ** 2
        var_d_list.append(var_d)
        std_d = var_d ** (1 / 2)
        std_d_list.append(std_d)

    mu = 2*N*U*p
    variance = mu
    sigma = math.sqrt(variance)

    x_r0 = np.linspace(d1_distribution_list_min_r0, d1_distribution_list_max_r0, 200)
    x_r1 = np.linspace(d1_distribution_list_min_r1, d1_distribution_list_max_r1, 200)

    poisson_y = np.exp(-mu)*np.power(mu, x_r1)/scipy.special.factorial(x_r1)
    poisson_y_dpw = np.exp(-average_d_list[1])*np.power(average_d_list[1], x_r1)/scipy.special.factorial(x_r1)

    gamma_y = np.zeros(len(x_r0))
    ii = 0
    N2 = N
    for t in x_r0:
        print("t", t)
        t = t/(N*U*p)
        for i in range(2, N2):
            temp_jj = 1.0
            for j in [jt for jt in range(2, N2) if jt != i]:
                temp_jj *= scipy.special.binom(j, 2)/(scipy.special.binom(j, 2)-scipy.special.binom(i, 2))
            gamma_y[ii] += 1.0/(N*U*p) * scipy.special.binom(i, 2) * math.exp(-scipy.special.binom(i, 2) * t)*temp_jj
        ii += 1
    print("mu", mu)
    print("sigma", sigma)


plt.figure(1, figsize=(5, 3.5))
plt.title("$N={}$".format(N) + ", $U={}$".format(U) + ", $p={}$".format(p))

for i in range(len(rList)):
    d1_distribution_list[i][d1_distribution_list[i] == 0] = 'nan'


i=0
plt.plot(d1_distribution_list[i][0:np.max(np.where(d1_distribution_list[i]))+1], ".-", label= '$r={}$'.format(rList[i]), alpha=0.5)
for i in range(2, len(rList)):
    plt.plot(d1_distribution_list[i][:-1], ".-", label='$r={}$'.format(rList[i]), alpha=0.5)
i=1
plt.plot(d1_distribution_list[i][0:np.max(np.where(d1_distribution_list[i]))+1], ".-", label= '$r={}$'.format(rList[i]), alpha=0.5)

print(d1_distribution_list[i][0:np.max(np.where(d1_distribution_list[i]))+1])
if analytics == True:
    plt.plot(x_r0, gamma_y, label="$f_{h_{MRCA}}$")
    plt.plot(x_r1, poisson_y, label="$f_{P}$")  #(2NU)


plt.yscale("log")
plt.xlabel("Hamming distance $d$")
plt.ylabel("Frequency")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()




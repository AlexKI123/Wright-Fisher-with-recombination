import numpy
import numpy as np
from modules import selrec_r_U_cluster
import timeit
import matplotlib.pyplot as plt



N = 100
Umin = 0.0001
Umax = 1.0
r = 0.0
generations_a = 1000
datapoints = 20
p = 1.0
log = True
load = False
save = False


if load is True:
    save = False

print("N", N)
print("r", r)
print("gen", generations_a)
print("\n")

startfull = timeit.default_timer()

if load is False:
    if log is False:
        Ulist = numpy.linspace(Umin, Umax, datapoints)
    if log is True:
        Ulist_temp = numpy.linspace(numpy.log10(Umin), numpy.log10(Umax), datapoints)
        Ulist = numpy.power(10, Ulist_temp)

    rate_array = numpy.zeros((datapoints, 13))

    for x1 in range(datapoints):
        print("{}/{}".format(x1+1, datapoints))
        U = Ulist[x1]
        print("U=", round(U, 4))
        start1 = timeit.default_timer()

        #generations = generations_a
        #discarded_elements = discarded_elements_a
        generations = int((generations_a/U)**(1/1.3))
        discarded_elements = int(generations/10)

        print("generations", generations)
        print("discarded_elements", discarded_elements)


        rate_array[x1, 0] = r
        rate_array[x1, 1] = U


        temp_values = selrec_r_U_cluster(r, U, N, generations, discarded_elements, p)

        rate_array[x1, 2] = temp_values[0]
        rate_array[x1, 3] = temp_values[1]
        rate_array[x1, 4] = temp_values[2]
        rate_array[x1, 5] = temp_values[3]
        rate_array[x1, 6] = temp_values[4]
        rate_array[x1, 7] = temp_values[5]
        rate_array[x1, 8] = temp_values[6]
        rate_array[x1, 9] = temp_values[7]
        rate_array[x1, 10] = temp_values[8]
        rate_array[x1, 11] = temp_values[9]
        rate_array[x1, 12] = temp_values[10]


        stop1 = timeit.default_timer()
        print("Time={}s".format(round(stop1 - start1,2)))
        print("\n")


print(rate_array.shape)
stopfull = timeit.default_timer()
print("time", stopfull-startfull)


X = rate_array[:, 0]   #r
Y = rate_array[:, 1]   #U
Z2 = rate_array[:, 2]   #rate
Z3 = rate_array[:, 3]  #rate2
Z4 = rate_array[:, 4]  #rate_viable
Z5 = rate_array[:, 5]  #rate2_viable
Z6 = rate_array[:, 6]  #rate_fixed
Z7 = rate_array[:, 7]  #AvgD
Z8 = rate_array[:, 8]  #MaxD
Z9 = rate_array[:, 9]  #dist_ge
Z10 = rate_array[:, 10]  #dist_viable_ge
Z11 = rate_array[:, 11]  #seg_mut
Z12 = rate_array[:, 12]  #mean_F



plt.figure(1)
plt.plot(Y, p*N*Y)
plt.plot(Y, Z2, ".")
plt.plot(Y, Z5, ".")
plt.title("Rate: $p={}$, $r={}$".format(p, r))
plt.xlabel("Mutation rate $U$")
if log is True:
    plt.yscale("log")
    plt.xscale("log")


plt.figure(2)
urange = np.logspace(np.log10(Umin), np.log10(Umax), num=100)
plt.plot(urange, urange * N * 2 * sum([1.0 / ii for ii in range(1, N)]), label="$\overline{S}$")
plt.plot(Y, Z11, ".")
plt.title("# Seg. Mut.: $p={}$, $r={}$".format(p, r))
plt.xlabel("Mutation rate $U$")
if log is True:
    plt.yscale("log")
    plt.xscale("log")


plt.figure(3)
plt.plot(Y, Z9, ".")
plt.title("# Genotypes: $p={}$, $r={}$".format(p, r))
plt.plot(urange, sum([2 * N * urange / (2 * N * urange + x) for x in range(N)]), label="$\overline{Y}(U,r=0)$")
plt.xlabel("Mutation rate $U$")
if log is True:
    plt.yscale("log")
    plt.xscale("log")


plt.figure(4)
plt.plot(Y, Z7, ".")
plt.title("Mean hd: $p={}$, $r={}$".format(p, r))
plt.plot(urange, 2 * N * urange, label="$\overline{d}$")
plt.xlabel("Mutation rate $U$")
if log is True:
    plt.yscale("log")
    plt.xscale("log")

plt.show()





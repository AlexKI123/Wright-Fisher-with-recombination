import numpy
import numpy as np
from modules import selrec_r_U_cluster
import timeit
import matplotlib.pyplot as plt


N = 100
rmin = 0.01
rmax = 1.0
U = 0.1
generations = 10000
discarded_elements = int(generations/10)
datapoints = 100
model = 7
p = 0.7
log = True
load = False
save = True


if load is True:
    save = False

print("N", N)
print("U", U)
print("gen", generations)
print("model", model)
print("\n")

startfull = timeit.default_timer()

if load is False:
    if log is False:
        rlist = numpy.linspace(rmin, rmax, datapoints)
    if log is True:
        rlist_temp = numpy.linspace(numpy.log10(rmin), numpy.log10(rmax), datapoints)
        rlist = numpy.power(10, rlist_temp)
    rate_array = numpy.zeros((datapoints, 13))

    for x1 in range(datapoints):
        print("{}/{}".format(x1+1, datapoints))
        r = rlist[x1]
        print("U=", round(U, 4))
        start1 = timeit.default_timer()


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

else:
    rate_array = numpy.load('r_raw_afterComments/N{}_U{}_dp{}_gen{}_p{}.npy'.format(N, U, datapoints, generations, p))

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

if save is True and load is False:
    numpy.save('r_raw_afterComments/N{}_U{}_dp{}_gen{}_p{}.npy'.format(N, U, datapoints, generations, p), rate_array)



plt.figure(1)
plt.plot(X, p*N*Y)
plt.plot(X, Z4, ".")
plt.plot(X, Z5, ".")
plt.title("Rate: $p={}$, $U={}$".format(p, U))
plt.xlabel("Recombination rate $r$")
if log is True:
    #plt.yscale("log")
    plt.xscale("log")


plt.figure(2)
rrange = np.logspace(np.log10(rmin), np.log10(rmax), num=100)
plt.plot(rrange, [U * N * 2 * sum([1.0 / ii for ii in range(1, N)])]*len(rrange), label="$\overline{S}$")
plt.plot(X, Z11, ".")
plt.title("# Seg. Mut.: $p={}$, $U={}$".format(p, U))
plt.xlabel("Recombination rate $r$")
if log is True:
    #plt.yscale("log")
    plt.xscale("log")


plt.figure(3)
plt.plot(X, Z9, ".")
plt.title("# Genotypes: $p={}$, $U={}$".format(p, U))
plt.plot(rrange, [sum([2 * N * U / (2 * N * U + x) for x in range(N)])]*len(rrange), label="$\overline{Y}(U,r=0)$")
plt.xlabel("Recombination rate $r$")
if log is True:
    #plt.yscale("log")
    plt.xscale("log")


plt.figure(4)
plt.plot(X, Z7, ".")
plt.title("Mean hd: $p={}$, $U={}$".format(p, U))
plt.plot(rrange, [2 * N * U]*len(rrange), label="$\overline{d}$")
plt.xlabel("Recombination rate $r$")
if log is True:
    plt.yscale("log")
    plt.xscale("log")

plt.figure(5)
plt.plot(X, Z12, ".")
plt.title("Mean fitness: $p={}$, $U={}$".format(p, U))
plt.plot(rrange, [2 * N * U]*len(rrange), label="$\overline{w}$")
plt.xlabel("Recombination rate $r$")
if log is True:
    #plt.yscale("log")
    plt.xscale("log")

plt.show()





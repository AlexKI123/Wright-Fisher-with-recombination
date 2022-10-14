import numpy
import matplotlib.pyplot as plt
import os
import numpy as np


#################################################################################
#################################################################################
# Code to plot figure A1
#################################################################################
#################################################################################

plt.rcParams.update({'font.size': 12})
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)

N = 100
Umin = 0.001
Umax = 1.0
rmin = 0.001
rmax = 1.0
p = 1.0
generations = 1000000000
datapoints = 30
log = True


D2_Datapoints = int(datapoints*datapoints)
if log is False:
    rlist = numpy.linspace(rmin, rmax, datapoints)
    Ulist = numpy.linspace(Umin, Umax, datapoints)
if log is True:
    rlist_temp = numpy.linspace(numpy.log10(rmin), numpy.log10(rmax), datapoints)
    rlist = numpy.power(10, rlist_temp)
    Ulist_temp = numpy.linspace(numpy.log10(Umin), numpy.log10(Umax), datapoints)
    Ulist = numpy.power(10, Ulist_temp)
Ulist = numpy.flip(Ulist)

tm_array = numpy.load('saved_data/sel_rec_U_r_N{}_dp{}_rmin{}_rmax{}_Umin{}_Umax{}_gen{}_p{}.npy'.format(N, datapoints, rmin, rmax, Umin, Umax, generations, p))
plotno=7


Uline= 0.09236708571873861
UlineList_r = []
UlineList_Z = []
for i in range(len(tm_array[:, 1])):
    print(tm_array[i, 1])
    if tm_array[i, 1] == Uline:
        UlineList_r.append(tm_array[i, 0])
        UlineList_Z.append(tm_array[i, plotno])

Uline001= 0.010826367338740546
UlineList_r001 = []
UlineList_Z001 = []
for i in range(len(tm_array[:, 1])):
    print(tm_array[i, 1])
    if tm_array[i, 1] == Uline001:
        UlineList_r001.append(tm_array[i, 0])
        UlineList_Z001.append(tm_array[i, plotno])

Uline0001= 0.001
UlineList_r0001 = []
UlineList_Z0001 = []
for i in range(len(tm_array[:, 1])):
    print(tm_array[i, 1])
    if tm_array[i, 1] == Uline0001:
        UlineList_r0001.append(tm_array[i, 0])
        UlineList_Z0001.append(tm_array[i, plotno])


fig, axs = plt.subplots(1, 2, figsize=(7,4), sharey=True)

axs[0].plot(np.array(UlineList_r), np.array(UlineList_Z), ".", color="C0")
axs[0].plot(np.array(UlineList_r001), np.array(UlineList_Z001), ".", color="C1")
axs[0].plot(np.array(UlineList_r0001), np.array(UlineList_Z0001), ".", color="C2")

axs[0].plot(np.array(UlineList_r), 2.0*N*Uline/(1.0+np.array(UlineList_r)*(2.0-np.array(UlineList_r))), color="C0", linewidth=2.0)
axs[0].plot(np.array(UlineList_r001), 2.0*N*Uline001/(1.0+np.array(UlineList_r001)*(2.0-np.array(UlineList_r001))), color="C1", linewidth=2.0)
axs[0].plot(np.array(UlineList_r0001), 2.0*N*Uline0001/(1.0+np.array(UlineList_r0001)*(2.0-np.array(UlineList_r0001))), color="C2", linewidth=2.0)


plotno=11


UlineList_Z = []
for i in range(len(tm_array[:, 1])):
    print(tm_array[i, 1])
    if tm_array[i, 1] == Uline:
        UlineList_Z.append(tm_array[i, plotno])

UlineList_Z001 = []
for i in range(len(tm_array[:, 1])):
    print(tm_array[i, 1])
    if tm_array[i, 1] == Uline001:
        UlineList_Z001.append(tm_array[i, plotno])

UlineList_Z0001 = []
for i in range(len(tm_array[:, 1])):
    print(tm_array[i, 1])
    if tm_array[i, 1] == Uline0001:
        UlineList_Z0001.append(tm_array[i, plotno])


axs[1].plot(np.array(UlineList_r), np.array(UlineList_Z), ".", color="C0", label="U=0.1")
axs[1].plot(np.array(UlineList_r001), np.array(UlineList_Z001), ".", color="C1", label="U=0.01")
axs[1].plot(np.array(UlineList_r0001), np.array(UlineList_Z0001), ".", color="C2", label="U=0.001")

axs[1].plot(np.array(UlineList_r), 2.0*N*Uline/(1.0+np.array(UlineList_r)*(2.0-np.array(UlineList_r)))*sum([1.0/i for i in range(1, N)]), color="C0", linewidth=2, zorder=100)
axs[1].plot(np.array(UlineList_r001), 2.0*N*Uline001/(1.0+np.array(UlineList_r001)*(2.0-np.array(UlineList_r001)))*sum([1.0/i for i in range(1, N)]), color="C1", linewidth=2.0, zorder=100)
axs[1].plot(np.array(UlineList_r0001), 2.0*N*Uline0001/(1.0+np.array(UlineList_r0001)*(2.0-np.array(UlineList_r0001)))*sum([1.0/i for i in range(1, N)]), color="C2", linewidth=2.0, zorder=100)

axs[0].set_title("$\overline{d}_{pw}$")
axs[1].set_title("$\overline{S}$")

axs[0].set_yscale("log")
axs[0].set_xscale("log")
axs[1].set_yscale("log")
axs[1].set_xscale("log")

axs[0].set_xticks([0.001,0.01,0.1,1])
axs[1].set_xticks([0.001,0.01,0.1,1])

axs[0].set_xlabel("$r$")
axs[1].set_xlabel("$r$")


plt.legend(bbox_to_anchor=(1.0, 0.65))
plt.tight_layout()
plt.show()


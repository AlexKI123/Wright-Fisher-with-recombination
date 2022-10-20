import numpy
import matplotlib.pyplot as plt

#################################################################################
#################################################################################
#Code to plot Figs. 9, S2, S4, S5, S6_Left
#################################################################################
#################################################################################

figure = "S6_Left"

if figure == "S4" or figure == "S2_Left":
    U = 0.05
else:
    U = 0.1

N = 100  # 100
rmin = 0.01
rmax = 1.0
generations = 1_000_000


if figure == "S4":
    plist = [0.7, 0.7, 0.7, 0.7]
else:
    plist = [0.5, 0.7, 0.9, 1.0]

if figure=="S5_Right" or figure=="9" or figure == "S2_Right":
    datapointslist = [200, 200, 200, 200]
else:
    datapointslist=[100, 100,100, 100]

lethalfitness = [0, 2, 4, 6]

log = True

tm_array = []
for i in range(len(plist)):
    datapoints =  datapointslist[i]
    p = plist[i]
    if figure == "S6_Left":
        tm_array.append(numpy.load('saved_data/Poisson_r_OBL_Crossovers_N{}_dp{}_rmin{}_rmax{}_U{}_gen{}_p{}.npy'.format(N, datapoints, rmin, rmax, U, generations, p)))
    if figure == "S5_Left":
        tm_array.append(numpy.load('saved_data/Poisson_r_SC_N{}_dp{}_rmin{}_rmax{}_U{}_gen{}_p{}.npy'.format(N, datapoints, rmin, rmax, U, generations, p)))
    if figure == "S2_Left":
        tm_array.append(numpy.load('saved_data/Poisson_r_N{}_dp{}_rmin{}_rmax{}_U{}_gen{}_p{}.npy'.format(N, datapoints, rmin, rmax, U, generations, p)))
    if figure == "S5_Right" or figure == "9" or figure == "S2_Right":
        tm_array.append(numpy.load('saved_data/r_N{}_dp{}_rmin{}_rmax{}_U{}_gen{}_p{}.npy'.format(N, datapoints, rmin, rmax, U, generations, p)))
    if figure == "S4":
        tm_array.append(numpy.load('saved_data/Poisson_r_N{}_dp{}_rmin{}_rmax{}_U{}_gen{}_p{}_selectlethal0{}.npy'.format(N, datapoints, rmin, rmax, U, generations, p, lethalfitness[i])))
    #tm_array.append(numpy.load('saved_data/r_N{}_dp{}_rmin{}_rmax{}_U{}_gen{}_p{}.npy'.format(N, datapoints, rmin, rmax, U, generations, p)))
    #tm_array.append(numpy.load('saved_data/sel_rec_r_N{}_dp{}_rmin{}_rmax{}_U{}_gen{}_p{}.npy'.format(N, datapoints, rmin, rmax, U, generations, p)))


import matplotlib.gridspec as gridspec


fig = plt.figure(figsize=(3.9,10))
gs = gridspec.GridSpec(8, 1, left=0.2, wspace=0.6, hspace=0.18)


ax1 = plt.subplot(gs[0, :])
ax2 = plt.subplot(gs[1, :])
ax3 = plt.subplot(gs[2, :])
ax4 = plt.subplot(gs[3, :])
ax5 = plt.subplot(gs[4, :])
ax6 = plt.subplot(gs[5, :])
ax7 = plt.subplot(gs[6, :])


ax1.set_xscale("log")
ax2.set_xscale("log")
ax3.set_xscale("log")
ax4.set_xscale("log")
ax5.set_xscale("log")
ax6.set_xscale("log")
ax7.set_xscale("log")


ax1.axes.xaxis.set_ticklabels([])
ax2.axes.xaxis.set_ticklabels([])
ax3.axes.xaxis.set_ticklabels([])
ax4.axes.xaxis.set_ticklabels([])
ax5.axes.xaxis.set_ticklabels([])
ax6.axes.xaxis.set_ticklabels([])


if figure=="S6_Left":
    ax1.set_title("Obligate sexual repro.\n$U={}$".format(U), fontsize=15)
if figure=="S5_Left":
    ax1.set_title("One-point crossover\n$U={}$".format(U), fontsize=15)
if figure=="S5_Right":
    ax1.set_title("Uniform crossover\n$U={}$".format(U), fontsize=15)
if figure == "S4":
    ax1.set_title("$U={}$, $p={}$, $w_1={}$".format(U, p, 1.0), fontsize=15)


colorList= ["C2", "C1", "C3", "C9"]
for i in range(len(plist)):
    ax1.plot(tm_array[i][:,0], tm_array[i][:,2], ".", markersize=2, label="rate", alpha=0.5, color=colorList[i])
    ax2.plot(tm_array[i][:,0], tm_array[i][:,6], ".", markersize=2, label="# rate fixed", alpha=0.5, color=colorList[i])
    ax3.plot(tm_array[i][:,0], tm_array[i][:,9], ".", markersize=2, label="# genotypes", alpha=0.5, color=colorList[i])
    ax4.plot(tm_array[i][:,0], tm_array[i][:,11], ".", markersize=2, label="seg. mut.", alpha=0.5, color=colorList[i])
    ax5.plot(tm_array[i][:,0], tm_array[i][:,7], ".", markersize=2, label="dpw", alpha=0.5, color=colorList[i])
    ax6.plot(tm_array[i][:,0], tm_array[i][:,12], ".", markersize=2, label="mean fitness", alpha=0.5, color=colorList[i])
    ax7.plot(tm_array[i][:,0], tm_array[i][:,13], ".", markersize=2, label="p={}".format(plist[i]), alpha=0.5, color=colorList[i])


ax7.set_xlabel("recombination rate $r$", fontsize=14)

ax1.set_ylabel("$r_{dis}$", fontsize=14)
ax2.set_ylabel("$r_{fix}$", fontsize=14)
ax3.set_ylabel("$\overline{Y}$", fontsize=14)
ax4.set_ylabel("$\overline{S}$", fontsize=14)
ax5.set_ylabel("$\overline{d}_{pw}$", fontsize=14)
ax6.set_ylabel("$\overline{w}$", fontsize=14)
ax7.set_ylabel("$v$", fontsize=14)

fig.subplots_adjust(left=0.0)
fig.subplots_adjust(right=0.98)
fig.subplots_adjust(top=0.95)
fig.subplots_adjust(bottom=0.03)

fig.align_labels()
plt.show()






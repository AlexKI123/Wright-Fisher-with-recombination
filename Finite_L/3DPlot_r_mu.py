import numpy
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.special
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
#plt.rcParams.update({'font.size': 12})
#plt.matplotlib.rc('xtick', labelsize=11)
#plt.matplotlib.rc('ytick', labelsize=11)

plt.rcParams.update({'font.size': 12})
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)

#################################################################################
#################################################################################
# Code to plot Fig. 10, S7, S16
#################################################################################
#################################################################################


rmin = 0.001
rmax = 1.0
mumin= 1e-06  # if Fig. 10, S7: 1e-06    if Fig. S16: 0.0001
mumax= 0.5
datapoints = 30
avg = 10000
Min_mutations = 20000  # if Fig. 10, S7: 20000    if Fig. S16: 200000
Visited = 1
log = True
l = 10
N = 100
p = 0.5
k = 3
landscape = "perc2"
mufliped= True
logz = False


save = False
load = True

D2_Datapoints = int(datapoints*datapoints)
if load == False:
    if log is False:
        rlist = numpy.linspace(rmin, rmax, datapoints)
        mulist = numpy.linspace(mumin, mumax, datapoints)
    if log is True:
        rlist_temp = numpy.linspace(numpy.log10(rmin), numpy.log10(rmax), datapoints)
        rlist = numpy.power(10, rlist_temp)
        mulist_temp = numpy.linspace(numpy.log10(mumin), numpy.log10(mumax), datapoints)
        mulist = numpy.power(10, mulist_temp)

    if mufliped is True:
        mulist = numpy.flip(mulist)


    tm_array = numpy.zeros((0, 11))
    counter_points = 0
    for x1 in range(D2_Datapoints):
        if os.path.isfile('L{}_N{}_p{}_avg{}_dp{}_vi{}_mM{}_i{}.npy'.format(l, N, p, avg, datapoints, Visited, Min_mutations, int(x1))):
            counter_points += 1
            #print("x1", x1)
            tm_array = numpy.append(tm_array, numpy.load('L{}_N{}_p{}_avg{}_dp{}_vi{}_mM{}_i{}.npy'.format(l, N, p, avg, datapoints, Visited, Min_mutations, int(x1))), axis=0)
            awdwad = numpy.load('L{}_N{}_p{}_avg{}_dp{}_vi{}_mM{}_i{}.npy'.format(l, N, p, avg, datapoints, Visited, Min_mutations, int(x1)))
        else:
            if mufliped is False:
                r = rlist[x1 // datapoints]
                mu = mulist[x1 % datapoints]
            else:
                r = rlist[x1 % datapoints]
                mu = mulist[x1 // datapoints]
            array_temp = [numpy.array([r, mu, 15000, 10000, 0.5, 0.01, 0.01, 0.01, 0.6, 0.5, 0.1])]
            tm_array = numpy.append(tm_array, array_temp, axis=0)
    if save is True:
        numpy.save('saved_data/L{}_N{}_p{}_avg{}_dp{}_rmin{}_rmax{}_mumin{}_mumax{}_vi{}_mM{}'.format(l, N, p, avg, datapoints, rmin, rmax, mumin, mumax, Visited, Min_mutations), tm_array)
    print(counter_points)

else:
    # if Fig. 10, S7: 1e-06
    tm_array = numpy.load('saved_data/RMU_v3_L{}_N{}_p{}_avg{}_dp{}_rmin{}_rmax{}_mumin{}_mumax{}_vi{}_mM{}_{}.npy'.format(l, N, p, avg, datapoints, rmin, rmax, mumin, mumax, Visited, Min_mutations, landscape))

    # if Fig.S16: 0.0001
    #tm_array = numpy.load('saved_data/RMU_sel_rec_L{}_N{}_p{}_avg{}_dp{}_rmin{}_rmax{}_mumin{}_mumax{}_vi{}_mM{}_{}.npy'.format(l, N, p, avg, datapoints, rmin, rmax, mumin, mumax, Visited, Min_mutations, landscape))


print(tm_array.shape)


#2 gen_counter_avg
#3 mut_counter_avg
#4 m_value_avg
#5 avg_hd_avg
#6 nogenotypes_avg
#7 nosegmut_avg
#8 viable recombination fraction
#9 mean Fitness

plotno =4

Uline2=0.0331034122159882


#Uline= 0.010985277619674985
Uline= 0.01339184296186895
#Uline = 0.010985277619674985
#Uline=0.0010481067512269438




UlineList_r = []
UlineList_Z = []
Uline2List_Z = []
for i in range(len(tm_array[:, 1])):
    print(tm_array[i, 1])
    if tm_array[i, 1] == Uline:
        UlineList_r.append(tm_array[i, 0])
        UlineList_Z.append(tm_array[i, plotno])
        #UlineList_Z.append(tm_array[i, plotno]/(tm_array[i, 9]))
    if tm_array[i, 1] == Uline2:
        Uline2List_Z.append(tm_array[i, plotno])
"""
rline = 1.0
rlineList_U = []
rlineList_Z = []
for i in range(len(tm_array[:, 0])):
    #print(tm_array[i, 1])
    if tm_array[i, 0] == rline:
        rlineList_U.append(tm_array[i, 1])
        rlineList_Z.append(tm_array[i, plotno])

print(UlineList_Z[0])
print(UlineList_Z[-1])
"""


X=tm_array[:, 0]
Y=tm_array[:, 1]
#Z=tm_array[:, 2]
Z=tm_array[:, plotno]
#Z=(tm_array[:, plotno]/(tm_array[:, 9]))   #normalized mean robustness
#Z=tm_array[:, 8]/(tm_array[:, plotno]/(tm_array[:, 9]))   #normalized mean fitess

#Z=tm_array[:, plotno]*(tm_array[:, 9])    #number of distinct and viable genotypes




mumaxlim = 0.5
muminlim = 0.000001
if plotno == 4:
    muminlim = 0.000001
if plotno == 5:
    if logz is False:
        muminlim = 0.00008
    else:
        muminlim = 0.00008
if plotno == 6 or plotno == 10:
    if logz is True:
        muminlim = 0.00008
    else:
        muminlim = 0.00008
if plotno == 7:
    if logz is True:
        muminlim = 0.00008
    else:
        muminlim = 0.00008
if plotno == 9:
    if logz is True:
        muminlim = 0.00008
    else:
        muminlim = 0.00008

#muminlim = 0.00005

#print("len(Y)", len(Y))
mu_deleted = 0
X = numpy.delete(X, numpy.argwhere(Y < muminlim))
Z = numpy.delete(Z, numpy.argwhere(Y < muminlim))
Y = numpy.delete(Y, numpy.argwhere(Y < muminlim))

mu_deleted= int((D2_Datapoints-len(Y))/datapoints)
#print("len(Y)", mu_deleted)
mumin_effective = Y[-1]

x = numpy.reshape(X, (datapoints-mu_deleted, datapoints))
y = numpy.reshape(Y, (datapoints-mu_deleted, datapoints))
z = numpy.reshape(Z, (datapoints-mu_deleted, datapoints))
#print("x", x)
#print("y", y)
#print("z", z)

x = numpy.log10(x)
y = numpy.log10(y)
if logz == True:
    z = numpy.log10(z)

import matplotlib.ticker as mticker



#fig = plt.figure(figsize=(6, 8))
#fig = plt.figure(figsize=(5,5.3))
fig = plt.figure(figsize=(3.9,4.5))

ax = fig.add_subplot(111, projection='3d')
#new
ax.set_box_aspect((1, 1, 1.4))  # xy aspect ratio is 1:1, but stretches z axis


if log is True:
    #xticks = [1e1, 5*1e1, 1e2, 5*1e2, 1e3, 5*1e3, 1e4, 5*1e4]
    xtickslabels = [1e-3, 1e-2, 1e-1, 1e0]
    #xticks= numpy.linspace(numpy.log10(rmin), numpy.log10(rmax), 5)
    #xticks = numpy.power(10, xticks)
    xticks = numpy.log10(xtickslabels)
    xticks = numpy.round(xticks, 2)
    print("xticks", xticks)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtickslabels)

    if muminlim == 0.000001:
        ytickslabels = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 5 * 1e-1]
    elif muminlim < 0.00001:
        ytickslabels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 5 * 1e-1]
    else:
        ytickslabels = [1e-4, 1e-3, 1e-2, 1e-1, 5 * 1e-1]
    if plotno == 4 or plotno == 8:
        ytickslabels = [1e-6, 1e-4, 1e-2, 5 * 1e-1]
    #ytickslabels = [1e-3, 1e-2, 5 * 1e-1]

    #yticks = numpy.linspace(numpy.log10(mumin), numpy.log10(mumax), 5)
    #yticks = numpy.power(10, yticks)
    yticks = numpy.log10(ytickslabels)
    yticks = numpy.round(yticks, 2)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytickslabels)
    #ax.set_zlim3d(np.log10(1e1), np.log10(6*1e4))

    if plotno == 5 or plotno == 6 or plotno == 7 or plotno == 10:
        ztickslabel = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        zticks = numpy.log10(ztickslabel)
        zticks = numpy.round(zticks, 2)
        ax.set_zticks(zticks)
        ax.set_zticklabels(ztickslabel)

ax.set_xlabel('recombination rate $r$')
ax.set_ylabel('mutation rate $\mu$')
if plotno == 2:
    ax.set_zlabel('Log10[Generations]')
if plotno == 3:
    ax.set_zlabel('Log10[Total mutations]')
if plotno == 4:
    ax.set_zlabel('mutational robustness $\overline{m}$')
if plotno == 5:
    ax.set_zlabel('mean hamming distance $\overline{d}_{pw}$')
if plotno == 6 or plotno == 10:
    ax.set_zlabel('# distinct genotypes $\overline{Y}$')
if plotno == 7:
    ax.set_zlabel('# segregating mutations $\overline{S}$')
if plotno == 8:
    ax.set_zlabel('viable recombination fraction')
if plotno == 9:
    ax.set_zlabel('mean fitness')

#coolwarm
#surf = ax.plot_trisurf(numpy.log10(X),numpy.log10(Y),Z, alpha=0.8, cmap=cm.viridis)

#norm = plt.Normalize(z.min(), z.max())
#new
norm = plt.Normalize(z.min(), z.max()+(z.max()-z.min())/5)
colors = cm.plasma(norm(z))
rcount, ccount, _ = colors.shape

#surf = ax.plot_surface(x, y, z, alpha=0.8, cmap="viridis", facecolors=colors, shade=False)
surf = ax.plot_surface(x, y, z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False, zorder=1)
surf.set_facecolor((0,0,0,0))

#fig.colorbar(surf, shrink=0.5, aspect=5)
#ax.contour(x, y, z, 10, colors="black", linestyles="solid")
#ax.plot_wireframe(x, y, z, alpha=0.8)
#ax.set_zlim(ax.get_zlim()[0]-0, ax.get_zlim()[1])
#new3
print("ax.get_ylim()[0]",ax.get_ylim()[0])
if plotno == 4 or plotno == 5 or plotno == 6 or plotno ==7 or plotno == 10:
    ax.set_xlim(ax.get_xlim()[0], np.log10(rmax))
    ax.set_ylim(ax.get_ylim()[0]-0.2, np.log10(mumax)+0.01)
    #ax.set_xlim(np.log10(rmin), ax.get_xlim()[1])
    #ax.set_ylim(np.log10(mumin), ax.get_ylim()[1])
if plotno == 8:
    ax.set_xlim(ax.get_xlim()[0], np.log10(rmax))
    ax.set_ylim(np.log10(mumin),ax.get_ylim()[1])
if plotno == 9:
    ax.set_xlim(np.log10(rmin), ax.get_xlim()[1])
    ax.set_ylim(np.log10(muminlim),ax.get_ylim()[1])


if plotno == 6 or plotno == 10:
    if logz is False:
        ax.set_zlim(ax.get_zlim()[0] - (100 - ax.get_zlim()[0]) / 3, 100)
    else:
        ax.set_zlim(ax.get_zlim()[0] - (2 - ax.get_zlim()[0]) / 3, 2)
elif plotno == 4:
    #ax.set_zlim(ax.get_zlim()[0]-0.07, ax.get_zlim()[1])
    ax.set_zlim(ax.get_zlim()[0]-0.0, ax.get_zlim()[1])
elif plotno == 5 and logz is True:
    ax.set_zlim(ax.get_zlim()[0] - (1 - ax.get_zlim()[0]) / 3, 1)
elif plotno == 8:
    ax.set_zlim(ax.get_zlim()[0]-0.09, 1.0)
elif plotno == 9:
    ax.set_zlim(ax.get_zlim()[0]-(ax.get_zlim()[1]-ax.get_zlim()[0])/3, 1.0)
else:
    ax.set_zlim(ax.get_zlim()[0]-(ax.get_zlim()[1]-ax.get_zlim()[0])/3, ax.get_zlim()[1])

"""
if plotno == 5:
    if logz is False:
        ax.set_zlim(ax.get_zlim()[0]-1.8, 5)
if plotno == 6:
    if logz == False:
        ax.set_zlim(ax.get_zlim()[0] - 35, 100)
    if logz == True:
        ax.set_zlim(ax.get_zlim()[0]-0.8, 2)
if plotno == 7:
    if logz == True:
        pass
    else:
        ax.set_zlim(ax.get_zlim()[0] - 3, 10)
"""
#ax.set_zlim(0.5, 1)
#ax.contour(x, y, z, 20, cmap="viridis", linestyles="solid")
#ax.contour(x, y, z, 10, cmap="plasma", linestyles="solid", offset=ax.get_zlim()[0])
if plotno == 4:
    ax.contour(x, y, z,9, cmap="plasma", linestyles="solid", offset=ax.get_zlim()[0]-0.0075)
if plotno == 5:
    if logz is False:
        ax.contour(x, y, z, 7, cmap="plasma", linestyles="solid", offset=ax.get_zlim()[0]-0.15)
    else:
        ax.contour(x, y, z, 7, cmap="plasma", linestyles="solid", offset=ax.get_zlim()[0]-0.05)

if plotno == 6 or plotno == 10:
    if logz is False:
        ax.contour(x, y, z, 7, cmap="plasma", linestyles="solid", offset=ax.get_zlim()[0]-2.5)
    else:
        ax.contour(x, y, z, 7, cmap="plasma", linestyles="solid", offset=ax.get_zlim()[0]-0.05)
if plotno == 7:
    if logz == False:
        ax.contour(x, y, z, 5, cmap="plasma", linestyles="solid", offset=ax.get_zlim()[0]-0.25)
    else:
        if p == 0.5:
            ax.contour(x, y, z, 5, cmap="plasma", linestyles="solid", offset=ax.get_zlim()[0]-0.04)
        else:
            ax.contour(x, y, z, 5, cmap="plasma", linestyles="solid", offset=ax.get_zlim()[0] - 0.03)
if plotno == 8:
    ax.contour(x, y, z, 5, cmap="plasma", linestyles="solid", offset=ax.get_zlim()[0]-0.008)
if plotno == 9:
    ax.contour(x, y, z,6, cmap="plasma", linestyles="solid", offset=ax.get_zlim()[0]-0.015)

ax.text2D(0.4, 0.93, "$N$={}, $L$={}, $p$={}".format(N,l,p), transform=ax.transAxes)
"""
fig, ax = plt.subplots()
CS = ax.contour(x, y, z, 12)
ax.clabel(CS, inline=1, fontsize=12)
xtickslabels = [1e-3, 1e-2, 1e-1, 1e0]

xticks = numpy.log10(xtickslabels)
xticks = numpy.round(xticks, 3)
print("xticks", xticks)
ax.set_xticks(xticks)
ax.set_xticklabels(xtickslabels)

ytickslabels = [1e-3, 1e-2, 1e-1, 5*1e-1]

yticks = numpy.log10(ytickslabels)
yticks = numpy.round(yticks, 2)
ax.set_yticks(yticks)
ax.set_yticklabels(ytickslabels)
ax.set_xlabel('recombination rate $r$')
ax.set_ylabel('mutation rate $\mu$')
plt.title("$L$={}, $N$={}, $p$={}".format(l,N,p))
"""

if plotno == 5:
    if p == 1:
        murange = np.logspace(np.log10(mumin_effective), np.log10(mumax), num=100)
        rrange = np.logspace(np.log10(rmin), np.log10(rmin), num=100)
        if logz is False:
            ax.plot(np.log10(rrange), np.log10(murange), 2*(murange-1)*murange*l*N/(4*(murange-1)*murange*(N-1)-1), linewidth=3.0, color="C0", label="$\overline{d}_{pw}$", zorder=100)
        else:
            ax.plot(np.log10(rrange), np.log10(murange), np.log10(2*(murange-1)*murange*l*N/(4*(murange-1)*murange*(N-1)-1)), linewidth=3.0, color="C0", label="$\overline{d}_{pw}$", zorder=100)
            #ax.plot(np.log10(rrange), np.log10(murange), np.log10(p*2*l*N*(murange-1)*murange*(1+(p-1)*murange)/(-1+4*(murange-1)*murange*(-1+N+N*(p-1)*murange))), linewidth=3.0, color="C0", label="$\overline{d}_{pw}$", zorder=100)
            #ax.plot(np.log10(rrange), np.log10(murange), np.log10(p*2 * N * murange*l), linewidth=3.0, label="Inf L", zorder=100)
        rrange = np.logspace(np.log10(rmax), np.log10(rmax), num=100)


        #rrange = np.logspace(np.log10(0.1), np.log10(0.1), num=100)
    #ax.plot(np.log10(rrange), np.log10(murange), 2 * (murange - 1) * murange * l * N / (4 * (murange - 1) * murange * (N - 1) - 1), linewidth=3.0, color="C0", zorder=100)
    #rrange = np.logspace(np.log10(0.01), np.log10(0.01), num=100)
    #ax.plot(np.log10(rrange), np.log10(murange), 2 * (murange - 1) * murange * l * N / (4 * (murange - 1) * murange * (N - 1) - 1), linewidth=3.0, color="C0", zorder=100)
    #plt.legend(loc=(0.75, 0.75))
    #plt.legend(loc=(0.72, 0.7))
    if logz is True:
        if p == 1.0:
            plt.legend(loc=(0.72, 0.8))

if plotno == 7:
    murange = np.logspace(np.log10(mumin_effective), np.log10(0.0018), num=100)
    #murange = np.logspace(np.log10(mumin_effective), np.log10(0.001), num=100)

    rrange = np.logspace(np.log10(rmin), np.log10(rmin), num=100)
    #if logz is False:
    #    ax.plot(np.log10(rrange), np.log10(murange), 2*(murange-1)*murange*l*N/(4*(murange-1)*murange*(N-1)-1)*sum([1.0/i for i in range(1, N)]), linewidth=3.0, color="C0", zorder=100)
    #else:
        #ax.plot(np.log10(rrange), np.log10(murange), np.log10(2*(murange-1)*murange*l*N/(4*(murange-1)*murange*(N-1)-1)*sum([1.0/i for i in range(1, N)])), linewidth=3.0, color="C0", zorder=100)
    #    ax.plot(np.log10(rrange), np.log10(murange) ,np.log10(p*l*murange * N * 2 * sum([1.0 / ii for ii in range(1, N)])), color="C0", label="$p\overline{S}$", linewidth=3.0, zorder=100)
    #rrange = np.logspace(np.log10(rmax), np.log10(rmax), num=100)
    #if logz is False:
    #    ax.plot(np.log10(rrange), np.log10(murange), 2*(murange-1)*murange*l*N/(4*(murange-1)*murange*(N-1)-1)*sum([1.0/i for i in range(1, N)]), linewidth=3.0, color="C0", label="$\overline{S}$", zorder=100)
    #else:
    #    ax.plot(np.log10(rrange), np.log10(murange), np.log10(2*(murange-1)*murange*l*N/(4*(murange-1)*murange*(N-1)-1)*sum([1.0/i for i in range(1, N)])), linewidth=3.0, color="C0", label="$\overline{S}$", zorder=100)
    #plt.legend(loc=(0.77, 0.82))


if logz is False:
    if p==0.5:
        ax.plot(np.log10(np.array(UlineList_r)), np.log10(np.array([Uline]*len(UlineList_r))), np.array(UlineList_Z), color="C2", linewidth=3.0, zorder=20)
        #ax.plot(np.log10(np.array(UlineList_r)), np.log10(np.array([Uline2]*len(UlineList_r))), np.array(Uline2List_Z), color="C10", linewidth=3.0, zorder=20)
    if p==1:
        ax.plot(np.log10(np.array(UlineList_r)), np.log10(np.array([Uline]*len(UlineList_r))), np.array(UlineList_Z), color="C2", linewidth=3.0, zorder=100)
else:
    if p==0.5:
        ax.plot(np.log10(np.array(UlineList_r)), np.log10(np.array([Uline]*len(UlineList_r))), np.log10(np.array(UlineList_Z)), color="C2", linewidth=3.0, zorder=20)
    if p==1:
        ax.plot(np.log10(np.array(UlineList_r)), np.log10(np.array([Uline]*len(UlineList_r))), np.log10(np.array(UlineList_Z)), color="C2", linewidth=3.0, zorder=100)


#plt.tight_layout()
#fig.subplots_adjust(left=-0.01)
#fig.subplots_adjust(right=1.01)
#fig.subplots_adjust(top=1.2)
#fig.subplots_adjust(bottom=-0.1)

ax.view_init(15.52937, -118.94109)
fig.subplots_adjust(left=-0.05)
fig.subplots_adjust(right=1.05)
fig.subplots_adjust(top=1.2)
fig.subplots_adjust(bottom=-0.08)

if plotno == 4 or plotno == 9:
    '''
    ax.view_init(15.52937, 61.0589)
    fig.subplots_adjust(left=-0.02)
    fig.subplots_adjust(right=1.07)
    fig.subplots_adjust(top=1.2)
    fig.subplots_adjust(bottom=-0.08)
    '''

    ax.view_init(15.52937, -118.94109)
    fig.subplots_adjust(left=-0.05)
    fig.subplots_adjust(right=1.05)
    fig.subplots_adjust(top=1.2)
    fig.subplots_adjust(bottom=-0.08)

if plotno == 8:
    ax.view_init(15.52937, 118.94109)
    fig.subplots_adjust(left=-0.1)
    fig.subplots_adjust(right=1.0)
    fig.subplots_adjust(top=1.2)
    fig.subplots_adjust(bottom=-0.08)

if plotno == 4:
    plt.savefig('FSM_p0.5_m_{}.pdf'.format(landscape))
if plotno == 5:
    plt.savefig('FSM_p{}_dpw_{}.pdf'.format(p, landscape))
if plotno == 6:
    plt.savefig('FSM_p{}_DisGen_{}.pdf'.format(p, landscape))
if plotno == 7:
    plt.savefig('FSM_p{}_SegMut_{}.pdf'.format(p, landscape))
if plotno == 8:
    plt.savefig('FSM_p{}_VRF_{}.pdf'.format(p, landscape))
if plotno == 9:
    plt.savefig('FSM_p{}_mF_{}.pdf'.format(p, landscape))
if plotno == 10:
    plt.savefig('FSM_p{}_DisGenVia_{}.pdf'.format(p, landscape))

plt.show()
print('ax.azim {}'.format(ax.azim))
print('ax.elev {}'.format(ax.elev))

"""
Standart:
ax.view_init(15.52937, -118.94109) 

An R-Achse gespiegelt:
ax.view_init(15.52937, -61.0589) 

Um 180 Grad gedreht:
ax.view_init(15.52937, 61.0589)
"""

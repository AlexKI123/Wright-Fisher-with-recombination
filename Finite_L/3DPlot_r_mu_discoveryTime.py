import numpy
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.special
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

plt.rcParams.update({'font.size': 12})
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

#################################################################################
#################################################################################
# Code to plot Fig. 11, S9
#################################################################################
#################################################################################

rmin = 0.001
rmax = 1.0
mumin= 0.000001
mumax= 0.5
datapoints = 30
avg = 1000  # if p=0.5 : 1000   if p=1.0 : 10000
log = True
l = 10
N = 100
p = 0.5
k = 3
landscape = "perc2"

mufliped= True

save = False
load = True

D2_Datapoints = int(datapoints*datapoints)
if load is False:
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

    tm_array = numpy.zeros((0, 8))
    counter_points = 0
    for x1 in range(D2_Datapoints):
        if os.path.isfile('L{}_N{}_p{}_avg{}_dp{}_i{}_s_im.npy'.format(l, N, p, avg, datapoints, int(x1))):
            counter_points += 1
            tm_array = numpy.append(tm_array, numpy.load('L{}_N{}_p{}_avg{}_dp{}_i{}_s_im.npy'.format(l, N, p, avg, datapoints, int(x1))),axis=0)
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
        numpy.save('saved_data/TM_L{}_N{}_p{}_avg{}_dp{}_rmin{}_rmax{}_mumin{}_mumax{}'.format(l, N, p, avg, datapoints, rmin, rmax, mumin, mumax), tm_array)
    print("counter_points", counter_points)

else:
    tm_array = numpy.load('saved_data/Ti_L{}_N{}_p{}_avg{}_dp{}_rmin{}_rmax{}_mumin{}_mumax{}_{}.npy'.format(l, N, p, avg, datapoints, rmin, rmax, mumin, mumax, landscape))

# 2 gen_counter_avg
# 3 mut_counter_avg
plotno = 3

Uline= 0.01339184296186895  # 0.01339184296186895
UlineList_r = []
UlineList_Z = []
for i in range(len(tm_array[:, 1])):
    print(tm_array[i, 1])
    if tm_array[i, 1] == Uline:
        UlineList_r.append(tm_array[i, 0])
        UlineList_Z.append(tm_array[i, plotno])

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




X=tm_array[:, 0]  #r
Y=tm_array[:, 1]  #U
Z=1.0/tm_array[:, plotno]
#Z=tm_array[:, plotno]

#for i in range(len(tm_array[:, 0])):
#    Z[i] = Z[i]*(N*l*Y[i])

#if plotno == 3:
#    Z=tm_array[:, plotno]
#Z=1.0/tm_array[:, plotno]
#print("Y",Y)


mumaxlim = 0.5
if plotno == 3:
    muminlim = 0.000001
if plotno == 2:
    muminlim = 0.000001
mu_deleted = 0
X = numpy.delete(X, numpy.argwhere(Y < muminlim))
Z = numpy.delete(Z, numpy.argwhere(Y < muminlim))
Y = numpy.delete(Y, numpy.argwhere(Y < muminlim))

X = numpy.delete(X, numpy.argwhere(Y>mumaxlim))
Z = numpy.delete(Z, numpy.argwhere(Y>mumaxlim))
Y = numpy.delete(Y, numpy.argwhere(Y>mumaxlim))

mu_deleted= int((D2_Datapoints-len(Y))/datapoints)
#print("len(Y)", mu_deleted)


x = numpy.reshape(X, (datapoints-mu_deleted, datapoints))
y = numpy.reshape(Y, (datapoints-mu_deleted, datapoints))
z = numpy.reshape(Z, (datapoints-mu_deleted, datapoints))
#print("y", y)
#print("z", z)

x = numpy.log10(x)
y = numpy.log10(y)
if plotno == 2 or plotno == 3:
    z = numpy.log10(z)

import matplotlib.ticker as mticker


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
    #print("xticks", xticks)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtickslabels)

    if muminlim == 0.000001:
        ytickslabels = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 5*1e-1]
    elif muminlim == 0.00001:
        ytickslabels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 5 * 1e-1]
    else:
        ytickslabels = [1e-4, 1e-3, 1e-2, 1e-1, 5 * 1e-1]
    ytickslabels = [1e-6, 1e-4, 1e-2, 5 * 1e-1]
    #ytickslabels = [1e-4, 1e-3, 1e-2, 1e-1, 5*1e-1]
    #yticks = numpy.linspace(numpy.log10(mumin), numpy.log10(mumax), 5)
    #yticks = numpy.power(10, yticks)
    yticks = numpy.log10(ytickslabels)
    yticks = numpy.round(yticks, 2)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytickslabels)
    #ax.set_zlim3d(np.log10(1e1), np.log10(6*1e4))


    ax.set_xlabel('recombination rate $r$')
ax.set_ylabel('mutation rate $\mu$')
if plotno == 2:
    ax.set_zlabel('Log10[1/Generations]')
if plotno == 3:
    ax.set_zlabel('Log10[1/Mutation events]')
if plotno == 4:
    ax.set_zlabel('mutational robustness $\overline{m}$')
if plotno == 5:
    ax.set_zlabel('mean hamming distance $\overline{d}_{pw}$')
if plotno == 6:
    ax.set_zlabel('# distinct genotypes $\overline{Y}$')
if plotno == 7:
    ax.set_zlabel('# segregating mutations $\overline{S}$')


#ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#coolwarm
#surf = ax.plot_trisurf(numpy.log10(X),numpy.log10(Y),Z, alpha=0.8, cmap=cm.viridis)

#norm = plt.Normalize(z.min(), z.max())
norm = plt.Normalize(z.min(), z.max()+(z.max()-z.min())/5)
colors = cm.plasma(norm(z))
rcount, ccount, _ = colors.shape

#surf = ax.plot_surface(x, y, z, alpha=0.8, cmap="viridis", facecolors=colors, shade=False)
surf = ax.plot_surface(x, y, z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
surf.set_facecolor((0,0,0,0))

#fig.colorbar(surf, shrink=0.5, aspect=5)
#ax.contour(x, y, z, 10, colors="black", linestyles="solid")
#ax.plot_wireframe(x, y, z, alpha=0.8)
#ax.set_zlim(ax.get_zlim()[0]-0.0, ax.get_zlim()[1])
#ax.set_ylim(numpy.log10(muminlim), numpy.log10(0.5))
#ax.contour(x, y, z, 20, cmap="viridis", linestyles="solid")
#ax.set_zlim(ax.get_zlim()[0]-(ax.get_zlim()[1]-ax.get_zlim()[0])/3, ax.get_zlim()[1])
if plotno == 2:
    ax.set_zlim(ax.get_zlim()[0]-(ax.get_zlim()[1]-ax.get_zlim()[0])/3, ax.get_zlim()[1])
if plotno == 3 and p==1:
    ax.set_zlim(ax.get_zlim()[0]-(ax.get_zlim()[1]-ax.get_zlim()[0])/3, ax.get_zlim()[1])
#if plotno == 3 and p== 0.5:
#    ax.set_zlim(ax.get_zlim()[0] +1, ax.get_zlim()[1])

#ax.set_xlim(np.log10(rmin), ax.get_xlim()[1])
#ax.set_ylim(ax.get_ylim()[0], np.log10(mumax))
ax.set_xlim(ax.get_xlim()[0], np.log10(rmax))
ax.set_ylim(ax.get_ylim()[0]-0.2, ax.get_ylim()[1])  #ax.get_ylim()[1]
#if plotno == 3:
#    ax.set_xlim(np.log10(rmin), ax.get_xlim()[1])
#    ax.set_ylim(np.log10(muminlim), ax.get_ylim()[1])

if plotno == 2:
    ax.contour(x, y, z, 7, cmap="plasma", linestyles="solid", offset=ax.get_zlim()[0]-0.2)
if plotno == 3:
    if p == 1:
        ax.contour(x, y, z, 10, cmap="plasma", linestyles="solid", offset=ax.get_zlim()[0]-0.04)
    if p == 0.5:
        ax.contour(x, y, z, 12, cmap="plasma", linestyles="solid", offset=ax.get_zlim()[0]-0.06)
if plotno == 6:
    ax.contour(x, y, z, 8, cmap="plasma", linestyles="solid", offset=ax.get_zlim()[0]-2.5)
if plotno == 7:
    ax.contour(x, y, z, 8, cmap="plasma", linestyles="solid", offset=ax.get_zlim()[0]-0.2)

ax.text2D(0.4, 0.93, "$N$={}, $L$={}, $p$={}".format(N, l, p), transform=ax.transAxes)


if plotno == 5:
    murange = np.logspace(np.log10(mumin), np.log10(mumax), num=100)
    rrange = np.logspace(np.log10(rmin), np.log10(rmin), num=100)
    ax.plot(np.log10(rrange), np.log10(murange), 2*(murange-1)*murange*l*N/(4*(murange-1)*murange*(N-1)-1), linewidth=3.0, color="C0", label="$\overline{d}_{pw}$", zorder=10)
    rrange = np.logspace(np.log10(rmax), np.log10(rmax), num=100)
    ax.plot(np.log10(rrange), np.log10(murange), 2*(murange-1)*murange*l*N/(4*(murange-1)*murange*(N-1)-1), linewidth=3.0, color="C0", zorder=10)
    rrange = np.logspace(np.log10(0.1), np.log10(0.1), num=100)
    ax.plot(np.log10(rrange), np.log10(murange), 2 * (murange - 1) * murange * l * N / (4 * (murange - 1) * murange * (N - 1) - 1), linewidth=3.0, color="C0", zorder=10)
    rrange = np.logspace(np.log10(0.01), np.log10(0.01), num=100)
    ax.plot(np.log10(rrange), np.log10(murange), 2 * (murange - 1) * murange * l * N / (4 * (murange - 1) * murange * (N - 1) - 1), linewidth=3.0, color="C0", zorder=10)
    #plt.legend(loc=(0.75, 0.75))
    plt.legend(loc=(0.7, 0.7))

if plotno == 7:
    murange = np.logspace(np.log10(mumin), np.log10(0.0015), num=100)
    rrange = np.logspace(np.log10(rmin), np.log10(rmin), num=100)
    ax.plot(np.log10(rrange), np.log10(murange), 2*(murange-1)*murange*l*N/(4*(murange-1)*murange*(N-1)-1)*sum([1.0/i for i in range(1, N)]), linewidth=3.0, color="C0", label="$\overline{S}$", zorder=10)
    rrange = np.logspace(np.log10(rmax), np.log10(rmax), num=100)
    ax.plot(np.log10(rrange), np.log10(murange), 2*(murange-1)*murange*l*N/(4*(murange-1)*murange*(N-1)-1)*sum([1.0/i for i in range(1, N)]), linewidth=3.0, color="C0", label="$\overline{S}$", zorder=10)
    #plt.legend(loc=(0.75, 0.75))


if p==0.5:
    if plotno == 2:
        ax.plot(np.log10(np.array(UlineList_r)), np.log10(np.array([Uline]*len(UlineList_r))), np.log10(1.0/np.array(UlineList_Z)), color="C2", linewidth=3.0, zorder=20)
    if plotno == 3:
        #ax.plot(np.log10(np.array(UlineList_r)), np.log10(np.array([Uline]*len(UlineList_r))), np.log10(np.array(UlineList_Z)), color="C2", linewidth=3.0, zorder=20)
        ax.plot(np.log10(np.array(UlineList_r)), np.log10(np.array([Uline]*len(UlineList_r))), np.log10(1.0/np.array(UlineList_Z)), color="C2", linewidth=3.0, zorder=20)

if p==1:
    ax.plot(np.log10(np.array(UlineList_r)), np.log10(np.array([Uline]*len(UlineList_r))), np.log10(1.0/np.array(UlineList_Z)), color="C2", linewidth=3.0, zorder=100)
    #ax.plot(np.log10(np.array(UlineList_r)), np.log10(np.array([Uline]*len(UlineList_r))), np.log10(np.array(UlineList_Z)), color="C9", linewidth=3.0, zorder=100)

#fig, ax = plt.subplots()
#CS = ax.contour(x, y, z, 12)
#ax.clabel(CS, inline=1, fontsize=12)
#xtickslabels = [1e-3, 1e-2, 1e-1, 1e0]

#xticks = numpy.log10(xtickslabels)
#xticks = numpy.round(xticks, 3)
#print("xticks", xticks)
#ax.set_xticks(xticks)
#ax.set_xticklabels(xtickslabels)

#ytickslabels = [1e-3, 1e-2, 1e-1, 5*1e-1]

#yticks = numpy.log10(ytickslabels)
#yticks = numpy.round(yticks, 2)
#ax.set_yticks(yticks)
#ax.set_yticklabels(ytickslabels)
#ax.set_xlabel('recombination rate $r$')
#ax.set_ylabel('mutation rate $\mu$')
#plt.title("$L$={}, $N$={}, $p$={}".format(l,N,p))


#plt.tight_layout()
#fig.subplots_adjust(left=-0.01)
#fig.subplots_adjust(right=1.01)
#fig.subplots_adjust(top=1.2)
#fig.subplots_adjust(bottom=-0.1)

fig.subplots_adjust(left=-0.05)
fig.subplots_adjust(right=1.05)
fig.subplots_adjust(top=1.2)
fig.subplots_adjust(bottom=-0.08)

if plotno == 3 and p==0.5:
    fig.subplots_adjust(left=-0.1)
    fig.subplots_adjust(right=1.0)
    fig.subplots_adjust(top=1.2)
    fig.subplots_adjust(bottom=-0.08)

#if plotno == 3:
#    fig.subplots_adjust(left=-0.02)
#    fig.subplots_adjust(right=1.07)
#    fig.subplots_adjust(top=1.2)
#    fig.subplots_adjust(bottom=-0.08)

ax.view_init(15.52937, -118.94109)
if plotno == 3 and p==0.5:
    #ax.view_init(15.52937, 47.11773)
    ax.view_init(15.52937, -62.5881)

if plotno == 3 and p==1 or p==0.5:
    ax.view_init(15.52937, 56.294)
    fig.subplots_adjust(left=-0.02)
    fig.subplots_adjust(right=1.07)
    fig.subplots_adjust(top=1.2)
    fig.subplots_adjust(bottom=-0.08)

if plotno == 2 or plotno == 3:
    fig.subplots_adjust(left=-0.05)
    fig.subplots_adjust(right=1.05)
    fig.subplots_adjust(top=1.2)
    fig.subplots_adjust(bottom=-0.08)
    ax.view_init(15.52937, -118.94109)

#    ax.view_init(15.52937, 56.294)

if plotno == 2:
    plt.savefig('FSM_p{}_Generations_v2_{}.pdf'.format(p, landscape))
if plotno == 3:
    plt.savefig('FSM_p{}_TM_{}.pdf'.format(p, landscape))

plt.show()
print('ax.azim {}'.format(ax.azim))
print('ax.elev {}'.format(ax.elev))
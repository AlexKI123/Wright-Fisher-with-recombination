import numpy
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

#################################################################################
#################################################################################
# Code to plot Figs. 2, 3, 4, 5, 6, 8, S3.
#################################################################################
#################################################################################

plt.rcParams.update({'font.size': 12})
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)

N = 100  # 1000 for S3 Fig.
Umin = 0.001
Umax = 1.0
rmin = 0.001
rmax = 1.0
p = 0.5  # 0.5 1.0
sel_rec = False  # simple successive recombination
if N == 100:
    if p == 1.0:
        generations = 10000000000
    if p == 0.5:
        generations = 10000000000
if N == 1000:
    generations = 1000000000
datapoints = 30
log = True
FigSave = True




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


if sel_rec is False:
    tm_array = numpy.load('saved_data/U_r_Poisson_N{}_dp{}_rmin{}_rmax{}_Umin{}_Umax{}_gen{}_p{}.npy'.format(N, datapoints, rmin, rmax, Umin, Umax, generations, p))
else:
    tm_array = numpy.load('saved_data/sel_rec_U_r_N{}_dp{}_rmin{}_rmax{}_Umin{}_Umax{}_gen{}_p{}.npy'.format(N, datapoints, rmin, rmax, Umin, Umax, generations, p))




# 2 discover rate
# 3 discover rate2
# 4 discover rate_viable---Fig 2.
# 5 discover rate2_viable
# 6 rate_fixed---Fig 3.
# 7 mean hamming distance---Fig 6.
# 8 max_dist
# 9 distinct genotypes
# 10 viable distinct genotypes---Fig.4
# 11 segregating mutations---Fig.5
# 12 mean_F---Fig. 8
# 13 viable fraction---Fig. 8
# 14 lethal fraction
# 15 novel fraction

plotno = 4


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

print(UlineList_Z[0])
print(UlineList_Z[-1])


X=tm_array[:, 0]    # r
Y=tm_array[:, 1]    # U
Z=tm_array[:, plotno]

x = numpy.reshape(X, (datapoints, datapoints))
y = numpy.reshape(Y, (datapoints, datapoints))
z = numpy.reshape(Z, (datapoints, datapoints))


if log is True:
    x = numpy.log10(x)
    y = numpy.log10(y)
    if plotno != 13 and plotno != 15 and plotno != 12:
        z = numpy.log10(z)


fig = plt.figure(figsize=(3.9,4.5))


ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((1, 1, 1.4))

if log is True:
    if rmin == 0.0001:
        xtickslabels = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    if rmin == 0.001:
        xtickslabels = [1e-3, 1e-2, 1e-1, 1e0]
    if rmin == 0.01:
        xtickslabels = [1e-2, 1e-1, 1e0]
    xticks = numpy.log10(xtickslabels)
    xticks = numpy.round(xticks, 2)
    print("xticks", xticks)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtickslabels)

    if Umin == 0.0001:
        ytickslabels = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    if Umin == 0.001:
        ytickslabels = [1e-3, 1e-2, 1e-1, 1e0]
    if Umin == 0.01:
        ytickslabels = [1e-2, 1e-1, 1e0]
    yticks = numpy.log10(ytickslabels)
    yticks = numpy.round(yticks, 2)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytickslabels)

    if plotno != 13 and plotno != 15 and plotno != 12:
        ztickslabel = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        zticks = numpy.log10(ztickslabel)
        zticks = numpy.round(zticks, 2)
        ax.set_zticks(zticks)
        ax.set_zticklabels(ztickslabel)


ax.set_xlabel('recombination rate $r$')
ax.set_ylabel('mutation rate $U$')
if plotno == 2 or plotno == 3 or plotno == 4 or plotno == 5:
    ax.set_zlabel('discovery rate')
if plotno == 6:
    ax.set_zlabel('rate fixed')
if plotno == 7:
    ax.set_zlabel('mean hamming distance $\overline{d}_{pw}$')
if plotno == 8:
    ax.set_zlabel('Log10[max hamming distance]')
if plotno == 9 or plotno == 10:
    ax.set_zlabel('# distinct genotypes $\overline{Y}$')
if plotno == 11:
    ax.set_zlabel('# segregating mutations $\overline{S}$')
if plotno == 12:
    ax.set_zlabel('mean Fitness')
if plotno == 13:
    ax.set_zlabel('viable recombiation fraction')
if plotno == 15:
    ax.set_zlabel('recombination novelty fraction')



norm = plt.Normalize(z.min(), z.max()+(z.max()-z.min())/5)
colors = cm.plasma(norm(z))
rcount, ccount, _ = colors.shape


surf = ax.plot_surface(x, y, z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
surf.set_facecolor((0,0,0,0))


if plotno ==13 or plotno == 15:
    zmax = 1
else:
    zmax = ax.get_zlim()[1]


if plotno == 6 or (plotno == 10 and p==0.5) or plotno==15:
    ax.set_xlim(np.log10(rmin), ax.get_xlim()[1])
    ax.set_ylim(ax.get_ylim()[0], np.log10(Umax))
else:
    if plotno == 13:
        ax.set_xlim(ax.get_xlim()[0], np.log10(rmax))
        ax.set_ylim(np.log10(Umin), ax.get_ylim()[1])
    if plotno == 12:
        ax.set_xlim(np.log10(rmin), ax.get_xlim()[1])
        ax.set_ylim(np.log10(Umin), ax.get_ylim()[1])
    else:
        ax.set_xlim(ax.get_xlim()[0], np.log10(rmax))
        ax.set_ylim(ax.get_ylim()[0], np.log10(Umax))


if plotno == 10 and p==0.5:
    ax.set_zlim(ax.get_zlim()[0]-(zmax-ax.get_zlim()[0])/3, np.log10(70))
else:
    ax.set_zlim(ax.get_zlim()[0]-(zmax-ax.get_zlim()[0])/3, zmax)


if plotno == 4 or plotno == 6 or plotno== 7 or plotno == 11:
    ax.contour(x, y, z, 6, cmap="plasma", linestyles="solid", offset=ax.get_zlim()[0]-0.08)
if plotno == 10:
    ax.contour(x, y, z, 6, cmap="plasma", linestyles="solid", offset=ax.get_zlim()[0]-0.05)
if plotno == 13 or plotno == 12:
    ax.contour(x, y, z, 6, cmap="plasma", linestyles="solid", offset=ax.get_zlim()[0]-0.012)
if plotno == 15:
    ax.contour(x, y, z, 6, cmap="plasma", linestyles="solid", offset=ax.get_zlim()[0]-0.03)


ax.text2D(0.4, 0.93, "$N$={}, $p$={}".format(N,p), transform=ax.transAxes)


if plotno == 7:
    Urange = np.logspace(np.log10(Umin), np.log10(Umax), num=100)
    rrange = np.logspace(np.log10(rmin), np.log10(rmin), num=100)
    if p==1:
        ax.plot(np.log10(rrange), np.log10(Urange), np.log10(2*N*Urange), linewidth=3.0, color="C0", label="$\overline{d}_{pw}$", zorder=100)
        rrange = np.logspace(np.log10(rmax), np.log10(rmax), num=100)
        plt.legend(loc=(0.7, 0.7))
    if p==0.5:
        ax.plot(np.log10(rrange), np.log10(Urange), np.log10(p*2*N*Urange), linewidth=3.0, color="C0", label="$p\overline{d}_{pw}$", zorder=100)
        rrange = np.logspace(np.log10(rmax), np.log10(rmax), num=100)


if plotno == 2 or plotno == 3:
    if log is True:
        urange = np.logspace(np.log10(Umin), np.log10(Umax), num=100)
        ax.plot([np.log10(rmin)] * len(urange), np.log10(urange), np.log10(N * urange), color="C0", linewidth=3.0, label="$NU$", zorder=100)
    else:
        urange = np.linspace(Umin, Umax, num=100)
        ax.plot([rmin]*len(urange), urange, N*urange, linewidth=3.0, label="$NU$")
    plt.legend(loc=(0.7, 0.7))


if plotno == 4 or plotno == 5:
    if log is True:
        urange = np.logspace(np.log10(Umin), np.log10(Umax), num=100)
        if p==1.0:
            ax.plot([np.log10(rmin)] * len(urange), np.log10(urange),np.log10(N * (1-numpy.exp(-urange))), linewidth=3.0, label="$r_{dis}$", zorder=100)
        else:
            ax.plot([np.log10(rmin)] * len(urange), np.log10(urange),np.log10(p*N * (1-numpy.exp(-urange))), linewidth=3.0, zorder=100)
    else:
        urange = np.linspace(Umin, Umax, num=100)
        ax.plot([rmin]*len(urange), urange, N*urange, linewidth=3.0, label="$NU$")
    if p == 1.0:
        plt.legend(loc=(0.1, 0.25),fontsize=14)

if plotno == 6:
    if log is True:
        urange = np.logspace(np.log10(Umin), np.log10(Umax), num=100)
        ax.plot([np.log10(rmin)] * len(urange), np.log10(urange), np.log10(N * urange * p / (N*numpy.exp(-urange)+N*p*(1-numpy.exp(-urange)))), linewidth=3.0, label="$r_{fix}$", zorder=100)

    if p== 1.0:
        plt.legend(loc=(0.17, 0.8))

if plotno == 9:
    if p == 1:
        urange = np.logspace(np.log10(Umin), np.log10(Umax), num=100)
        ax.plot([np.log10(Umin)] * len(urange), np.log10(urange),np.log10(sum([2 * N * urange / (2 * N * urange + x) for x in range(N)])), linewidth=3.0, label="$\overline{Y}$", zorder=100)
        plt.legend(loc=(0.73, 0.8))
    if p ==0.5:
        urange = np.logspace(np.log10(Umin), np.log10(Umax), num=100)
        ax.plot([np.log10(Umin)] * len(urange), np.log10(urange),np.log10(sum([2 * N * urange / (2 * N * urange + x) for x in range(N)])), linewidth=3.0, label="$\overline{Y}$", zorder=100)
        plt.legend(loc=(0.73, 0.8))


if plotno == 10:
    urange = np.logspace(np.log10(Umin), np.log10(Umax), num=100)
    ax.plot([np.log10(rmin)] * len(urange), np.log10(urange),np.log10(sum([p*2 * N *urange / (p*2 * N * urange + x) for x in range(N)])), linewidth=3.0, label="$\overline{Y}$", zorder=0)
    if p==1.0:
        plt.legend(loc=(0.75, 0.8))


if plotno == 11:
    if p==1:
        urange = np.logspace(np.log10(Umin), np.log10(Umax), num=100)
        ax.plot([np.log10(rmin)] * len(urange), np.log10(urange),np.log10(urange * N * 2 * sum([1.0 / ii for ii in range(1, N)])), color="C0", label="$\overline{S}$", linewidth=3.0, zorder=100)
        plt.legend(loc=(0.7, 0.7))
    if p==0.5:
        urange = np.logspace(np.log10(Umin), np.log10(Umax), num=100)
        ax.plot([np.log10(rmin)] * len(urange), np.log10(urange),np.log10(p*urange * N * 2 * sum([1.0 / ii for ii in range(1, N)])), color="C0", label="$p\overline{S}$", linewidth=3.0, zorder=100)



if plotno != 13 and plotno != 15 and plotno != 12:
    if p==0.5:
        ax.plot(np.log10(np.array(UlineList_r)), np.log10(np.array([Uline]*len(UlineList_r))), np.log10(np.array(UlineList_Z)), color="C2", linewidth=3.0, zorder=100)
    if p==1:
        ax.plot(np.log10(np.array(UlineList_r)), np.log10(np.array([Uline]*len(UlineList_r))), np.log10(np.array(UlineList_Z)), color="C2", linewidth=3.0, zorder=100)
else:
    ax.plot(np.log10(np.array(UlineList_r)), np.log10(np.array([Uline]*len(UlineList_r))), np.array(UlineList_Z), color="C2", linewidth=3.0, zorder=100)


fig.subplots_adjust(left=-0.05)
fig.subplots_adjust(right=1.05)
fig.subplots_adjust(top=1.2)
fig.subplots_adjust(bottom=-0.08)

if plotno == 6 or (plotno == 10 and p==0.5) or plotno == 13:
    fig.subplots_adjust(left=-0.1)
    fig.subplots_adjust(right=1.0)
    fig.subplots_adjust(top=1.2)
    fig.subplots_adjust(bottom=-0.08)


if plotno == 12:
    fig.subplots_adjust(left=-0.02)
    fig.subplots_adjust(right=1.07)
    fig.subplots_adjust(top=1.2)
    fig.subplots_adjust(bottom=-0.08)


ax.view_init(15.52937, -118.94109)
if plotno == 6 or plotno == 15 or (plotno==10 and p==0.5):
    ax.view_init(15.52937, -61.0589)
if plotno == 13:
    ax.view_init(15.52937, 118.94109)
if plotno == 12:
    ax.view_init(15.52937, 61.0589)

if FigSave is True:
    if plotno == 7:
        plt.savefig('ISM_p{}_dpw.pdf'.format(p))
    elif plotno == 6:
        plt.savefig('ISM_p{}_Fixed.pdf'.format(p))
    elif plotno == 4:
        plt.savefig('ISM_p{}_Rate.pdf'.format(p))
    elif plotno == 9:
        plt.savefig('ISM_p{}_Gen.pdf'.format(p))
    elif plotno == 10:
        plt.savefig('ISM_p{}_Gen_v2.pdf'.format(p))
    elif plotno == 11:
        plt.savefig('ISM_p{}_SegMut.pdf'.format(p))
    elif plotno == 12:
        plt.savefig('ISM_p{}_mF.pdf'.format(p))
    elif plotno == 13:
        plt.savefig('ISM_p{}_VRF.pdf'.format(p))
    elif plotno == 15:
        plt.savefig('ISM_p{}_RN.pdf'.format(p))

plt.show()
print('ax.azim {}'.format(ax.azim))
print('ax.elev {}'.format(ax.elev))
import numpy
import matplotlib.pyplot as plt
import os
import numpy as np

#################################################################################
#################################################################################
# Code to plot Fig. 9_Left
#################################################################################
#################################################################################

N = 100  # 100
r = 1.0
Umin = 0.01
Umax = 1.0
generations = 10000000000  #100000000 #1000000  #100000000
plist = [0.5,0.7, 0.9, 1.0]
#datapointslist=[400,200,400,200]
datapointslist=[200,200,200,200]
log = True

tm_array = []
for i in range(len(plist)):
    datapoints =  datapointslist[i]
    p = plist[i]
    #tm_array.append(numpy.load('saved_data/U_N{}_dp{}_Umin{}_Umax{}_r{}_gen{}_p{}.npy'.format(N, datapoints, Umin, Umax, r, generations, p)))
    tm_array.append(numpy.load('saved_data/Poisson_U_N{}_dp{}_Umin{}_Umax{}_r{}_gen{}_p{}.npy'.format(N, datapoints, Umin, Umax, r, generations, p)))




# 0 r
# 1 U
# 2 rate
# 3 rate2
# 4 rate_viable
# 5 rate2_viable
# 6 rate_fixed
# 7 avg_dist
# 8 max_dist
# 9 dist_ge
# 10 dist_viable_ge
# 11 seg_mut
# 12 mean_F

# 13 viable fraction
# 14 lethal fraction
# 15 novel fraction


#rate, rate2, rate_viable, rate2_viable, rate_fixed, avg_dist, max_dist, dist_ge, dist_viable_ge, seg_mut

#0 r
#1 U
#2 rate
#3 rate2
#4 rate_viable
#5 rate2_viable
#6 rate_fixed
#7 AvgD
#8 MaxD
#9 dist_ge
#10 dist_viable_ge
#11 seg_mut
#12 mean_F
#13 viable fraction
#14 lethal fraction
#15 novel fraction






import matplotlib.gridspec as gridspec
#fig = plt.figure(figsize=(4.3,8))
#gs = gridspec.GridSpec(7, 1, hspace=0.1)


fig = plt.figure(figsize=(3.9,10))
gs = gridspec.GridSpec(8, 1, left=0.2, wspace=0.6, hspace=0.18)


ax1 = plt.subplot(gs[0, :])
ax2 = plt.subplot(gs[1, :])
ax3 = plt.subplot(gs[2, :])
ax4 = plt.subplot(gs[3, :])
ax5 = plt.subplot(gs[4, :])
ax6 = plt.subplot(gs[5, :])
ax7 = plt.subplot(gs[6, :])
#ax8 = plt.subplot(gs[7, :])
#ax9 = plt.subplot(gs[8, :])

ax1.set_xscale("log")
ax2.set_xscale("log")
ax3.set_xscale("log")
ax4.set_xscale("log")
ax5.set_xscale("log")
ax6.set_xscale("log")
ax7.set_xscale("log")
#ax8.set_xscale("log")
#ax9.set_xscale("log")

ax1.set_yscale("log")
ax2.set_yscale("log")
ax3.set_yscale("log")
ax4.set_yscale("log")
ax5.set_yscale("log")
#ax6.set_yscale("log")
#ax7.set_yscale("log")



ax1.axes.xaxis.set_ticklabels([])
ax2.axes.xaxis.set_ticklabels([])
ax3.axes.xaxis.set_ticklabels([])
ax4.axes.xaxis.set_ticklabels([])
ax5.axes.xaxis.set_ticklabels([])
ax6.axes.xaxis.set_ticklabels([])
#ax7.axes.xaxis.set_ticklabels([])
#ax8.axes.xaxis.set_ticklabels([])



#fig.suptitle("$p={}$, $r={}$".format(p, r))
ax1.set_title("$r={}$".format(r), fontsize=15)

colorList= ["C2", "C1", "C3", "C9"]
for i in range(len(plist)):
    ax1.plot(tm_array[i][:,1], tm_array[i][:,2], ".", markersize=2, label="rate", alpha=0.5, color=colorList[i])
    ax2.plot(tm_array[i][:,1], tm_array[i][:,6], ".", markersize=2, label="# rate fixed", alpha=0.5, color=colorList[i])
    ax3.plot(tm_array[i][:,1], tm_array[i][:,9], ".", markersize=2, label="# genotypes", alpha=0.5, color=colorList[i])
    ax4.plot(tm_array[i][:,1], tm_array[i][:,11], ".", markersize=2, label="seg. mut.", alpha=0.5, color=colorList[i])
    ax5.plot(tm_array[i][:,1], tm_array[i][:,7], ".", markersize=2, label="dpw", alpha=0.5, color=colorList[i])
    #ax6.plot(Y, Z10/Z9, ".", markersize=2, label="# viable genotypes/# genotypes")
    ax6.plot(tm_array[i][:,1], tm_array[i][:,12], ".", markersize=2, label="mean fitness", alpha=0.5, color=colorList[i])

    ax7.plot(tm_array[i][:,1], tm_array[i][:,13], ".", markersize=2, label="p={}".format(plist[i]), alpha=0.5, color=colorList[i])
    #ax8.plot(Y, Z15, ".", markersize=2, label="novel fraction")


ax7.set_xlabel("mutation rate $U$", fontsize=14)



ax1.set_ylabel("$r_{dis}$", fontsize=14)
ax2.set_ylabel("$r_{fix}$", fontsize=14)
ax3.set_ylabel("$\overline{Y}$", fontsize=14)
ax4.set_ylabel("$\overline{S}$", fontsize=14)
ax5.set_ylabel("$\overline{d}_{pw}$", fontsize=14)
ax6.set_ylabel("$\overline{w}$", fontsize=14)
ax7.set_ylabel("$v$", fontsize=14)
#ax7.legend(bbox_to_anchor=(-1.,- 1.), loc="lower center")

fig.subplots_adjust(left=0.0)
fig.subplots_adjust(right=0.98)
fig.subplots_adjust(top=0.95)
fig.subplots_adjust(bottom=0.03)


#ax7.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), markerscale=4., fancybox=True, shadow=False, ncol=2, fontsize=13)




##fig.subplots_adjust(left=0.05)
#fig.subplots_adjust(right=0.98)
#fig.subplots_adjust(top=0.97)
#fig.subplots_adjust(bottom=0.07)
#plt.tight_layout()


fig.align_labels()
#plt.savefig("CrossSectionU_v3.pdf")


plt.savefig('Poisson_ISM_CrossSection_r1.pdf')



plt.show()






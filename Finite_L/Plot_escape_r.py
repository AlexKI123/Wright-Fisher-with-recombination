import numpy
import matplotlib.pyplot as plt
import os
import numpy as np


#################################################################################
#################################################################################
# Code to plot Fig. S8
#################################################################################
#################################################################################


def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]


def mean(a):
    return sum(a) / len(a)


rmin = 0.001
rmax = 1.0
mu = 0.01
datapoints = 30
avg = 1000000
log = True
l = 10
N = 100
p = 0.8
landscape = "perc"

load = True
save = False


if log is False:
    rlist = numpy.linspace(rmin, rmax, datapoints)
if log is True:
    rlist_temp = numpy.linspace(numpy.log10(rmin), numpy.log10(rmax), datapoints)
    rlist = numpy.power(10, rlist_temp)


r_list = []
nestedGenList = []
nesteddisList = []
nestedreiList = []
nestedrevList = []


counter_points = 0
for x1 in range(datapoints):
    print('Escape/L{}_N{}_p{}_avg{}_dp{}_i{}_{}.npy'.format(l, N, p, avg, datapoints, int(x1), landscape))
    if os.path.isfile('Escape/L{}_N{}_p{}_avg{}_dp{}_i{}_{}.npy'.format(l, N, p, avg, datapoints, int(x1), landscape)):
        counter_points += 1
        temp = numpy.load('Escape/L{}_N{}_p{}_avg{}_dp{}_i{}_{}.npy'.format(l, N, p, avg, datapoints, int(x1), landscape), allow_pickle=True)
        r_list.append(temp[0])
        nestedGenList.append(temp[1])
        nesteddisList.append(temp[2])
        nestedreiList.append(temp[3])
        nestedrevList.append(temp[4])

        print("x1", x1)
    else:
        r = rlist[x1]
        print('not yet')
        quit()

print("counter_points", counter_points)


rei = 10/10
rev = 10/10
dis = 7
CondnestedGenList = []
CondnesteddisList = []
CondnestedreiList = []
CondnestedrevList = []


for i in range(datapoints):
    indicesrei = find_indices(nestedreiList[i], lambda e: e == rei)
    indicesrev = find_indices(nestedrevList[i], lambda e: e == rev)
    indicesdis = find_indices(nesteddisList[i], lambda e: e == dis)

    indices = indicesrei
    #indices = indicesrev
    #indices = indicesdis

    #indices = list(set(indicesrei)&set(indicesrev))


    CondnestedGenList.append([nestedGenList[i][j] for j in indices])
    CondnesteddisList.append([nesteddisList[i][j] for j in indices])
    CondnestedreiList.append([nestedreiList[i][j] for j in indices])
    CondnestedrevList.append([nestedrevList[i][j] for j in indices])

#print(CondnestedrevList)

avgGenList=np.array([sum(col) / float(len(col)) for col in CondnestedGenList])


plt.figure(1, figsize=(5, 2.7))
plt.ylabel("1/Generations")
plt.xlabel("Recombination rate $r$")
plt.title("$L$={}, $N$={}, $\mu$={}, $p={}$".format(l, N, mu, p))
plt.loglog(r_list, 1/np.mean(np.array(nestedGenList), axis=1), "--", label="All")
plt.loglog(r_list, 1/avgGenList, "--", label="$m_{inital}=1.0$")


rei = 6/10
rev = 6/10
dis = 7
CondnestedGenList = []
CondnesteddisList = []
CondnestedreiList = []
CondnestedrevList = []


for i in range(datapoints):
    indicesrei = find_indices(nestedreiList[i], lambda e: e == rei)
    indicesrev = find_indices(nestedrevList[i], lambda e: e == rev)
    indicesdis = find_indices(nesteddisList[i], lambda e: e == dis)

    indices = indicesrei
    #indices = indicesrev
    #indices = indicesdis

    #indices = list(set(indicesrei)&set(indicesrev))


    CondnestedGenList.append([nestedGenList[i][j] for j in indices])
    CondnesteddisList.append([nesteddisList[i][j] for j in indices])
    CondnestedreiList.append([nestedreiList[i][j] for j in indices])
    CondnestedrevList.append([nestedrevList[i][j] for j in indices])

#print(CondnestedrevList)

avgGenList=np.array([sum(col) / float(len(col)) for col in CondnestedGenList])
plt.loglog(r_list, 1/avgGenList, "--", label="$m_{inital}=0.6$")


plt.legend()
plt.tight_layout()
plt.show()
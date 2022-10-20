import numpy
import matplotlib.pyplot as plt
import os
import numpy as np


N = 100
U = 0.05
rmin = 0.01
rmax = 1.0
generations = 1000000
datapoints = 100
p = 0.7
log = True
save = False


if log is False:
    rlist = numpy.linspace(rmin, rmax, datapoints)
if log is True:
    rlist_temp = numpy.linspace(numpy.log10(rmin), numpy.log10(rmax), datapoints)
    rlist = numpy.power(10, rlist_temp)


#tm_array = numpy.zeros((0, 13))
tm_array = numpy.zeros((0, 17))
counter_points = 0
for x1 in range(datapoints):
    if os.path.isfile('N{}_dp{}_gen{}_p{}_i{}.npy'.format(N, datapoints, generations, p, int(x1))):
        print(numpy.shape(numpy.load('N{}_dp{}_gen{}_p{}_i{}.npy'.format(N, datapoints, generations, p, int(x1)))))
        counter_points += 1
        tm_array = numpy.append(tm_array, numpy.load('N{}_dp{}_gen{}_p{}_i{}.npy'.format(N, datapoints, generations, p, int(x1))), axis=0)
    else:
        r = rlist[x1]
        print("x1", x1)
        #array_temp = [numpy.array([r, U, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
        array_temp = [numpy.array([r, U, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0])]

        tm_array = numpy.append(tm_array, array_temp, axis=0)

if save is True:
    numpy.save('saved_data/r_N{}_dp{}_rmin{}_rmax{}_U{}_gen{}_p{}_selectlethal08.npy'.format(N, datapoints, rmin, rmax, U, generations, p), tm_array)

print("counter_points", counter_points)


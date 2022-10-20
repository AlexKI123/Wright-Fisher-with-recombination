import numpy
import matplotlib.pyplot as plt
import os
import numpy as np



N = 100
r = 0.0
Umin = 0.001
Umax = 1.0
generations = 1000000000
datapoints = 30
p = 0.3
log = True
save = False


if log is False:
    Ulist = numpy.linspace(Umin, Umax, datapoints)
if log is True:
    Ulist_temp = numpy.linspace(numpy.log10(Umin), numpy.log10(Umax), datapoints)
    Ulist = numpy.power(10, Ulist_temp)


#tm_array = numpy.zeros((0, 13))
tm_array = numpy.zeros((0, 17))
counter_points = 0
for x1 in range(datapoints):
    if os.path.isfile('N{}_r{}_dp{}_gen{}_p{}_i{}.npy'.format(N, r, datapoints, generations, p, int(x1))):
        counter_points += 1
        #print("x1", x1)
        tm_array = numpy.append(tm_array, numpy.load('N{}_r{}_dp{}_gen{}_p{}_i{}.npy'.format(N, r, datapoints, generations, p, int(x1))), axis=0)
    else:
        U = Ulist[x1]
        print("x1", x1)
        #array_temp = [numpy.array([r, U, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
        array_temp = [numpy.array([r, U, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
        tm_array = numpy.append(tm_array, array_temp, axis=0)
print("counter_points", counter_points)

if save is True:
    numpy.save('saved_data/Poisson_U_N{}_dp{}_Umin{}_Umax{}_r{}_gen{}_p{}'.format(N, datapoints, Umin, Umax, r, generations, p), tm_array)


import numpy
import os


N = 1000
Umin = 0.001
Umax = 1.0
rmin = 0.001
rmax = 1.0
generations = 1000000000
datapoints = 30
p = 0.5
log = True
save = True


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

# tm_array = numpy.zeros((0, 13))
#tm_array = numpy.zeros((0, 16))
tm_array = numpy.zeros((0, 17))
counter_points = 0
for x1 in range(D2_Datapoints):
    if os.path.isfile('N{}_dp{}_gen{}_p{}_i{}.npy'.format(N, datapoints, generations, p, int(x1))):
        print("x1", x1)
        counter_points += 1
        tm_array = numpy.append(tm_array, numpy.load('N{}_dp{}_gen{}_p{}_i{}.npy'.format(N, datapoints, generations, p, int(x1))), axis=0)
    else:
        r = rlist[x1 % datapoints]
        U = Ulist[x1 // datapoints]
        #print("x1", x1)
        #array_temp = [numpy.array([r, U, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
        array_temp = [numpy.array([r, U, 1000, 1000, 500, 500, 1, 0, 0, 1000, 500, 8000, 0, 0, 0, 0, 0])]
        tm_array = numpy.append(tm_array, array_temp, axis=0)
print("counter_points", counter_points)

if save is True:
    numpy.save('saved_data/U_r_Poisson_N{}_dp{}_rmin{}_rmax{}_Umin{}_Umax{}_gen{}_p{}.npy'.format(N, datapoints, rmin, rmax, Umin, Umax, generations, p), tm_array)

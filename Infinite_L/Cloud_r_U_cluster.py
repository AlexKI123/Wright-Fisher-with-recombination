import sys
import numpy
import timeit
from modules import selrec_r_U_cluster
from modules import sel_rec_r_U_cluster


N = 100
Umin = 0.001
Umax = 1.0
rmin = 0.001
rmax = 1.0
generations_a = 1000000000
datapoints = 30
p = 0.5
log = True

discarded_elements_a = int(generations_a/10)

x1 = int(sys.argv[1])


if log is False:
    rlist = numpy.linspace(rmin, rmax, datapoints)
    Ulist = numpy.linspace(Umin, Umax, datapoints)
if log is True:
    rlist_temp = numpy.linspace(numpy.log10(rmin), numpy.log10(rmax), datapoints)
    rlist = numpy.power(10, rlist_temp)
    Ulist_temp = numpy.linspace(numpy.log10(Umin), numpy.log10(Umax), datapoints)
    Ulist = numpy.power(10, Ulist_temp)


Ulist = numpy.flip(Ulist)
r = rlist[x1 % datapoints]
U = Ulist[x1 // datapoints]

print("r=", r)
print("U=", U)

generations = int((generations_a / U) ** (1 / 2))
discarded_elements = int(generations/10)
print("generations", generations)
print("discarded_elements", discarded_elements)

tm_array = numpy.zeros((1, 17))

tm_array[0, 0] = r
tm_array[0, 1] = U

start = timeit.default_timer()
temp_values = selrec_r_U_cluster(r, U, N, generations, discarded_elements, p)
#temp_values = sel_rec_r_U_cluster(r, U, N, generations, discarded_elements, p)

tm_array[0, 2] = temp_values[0]
tm_array[0, 3] = temp_values[1]
tm_array[0, 4] = temp_values[2]
tm_array[0, 5] = temp_values[3]
tm_array[0, 6] = temp_values[4]
tm_array[0, 7] = temp_values[5]
tm_array[0, 8] = temp_values[6]
tm_array[0, 9] = temp_values[7]
tm_array[0, 10] = temp_values[8]
tm_array[0, 11] = temp_values[9]
tm_array[0, 12] = temp_values[10]
tm_array[0, 13] = temp_values[11]
tm_array[0, 14] = temp_values[12]
tm_array[0, 15] = temp_values[13]
tm_array[0, 16] = temp_values[14]

numpy.save('N{}_dp{}_gen{}_p{}_i{}.npy'.format(N, datapoints, generations_a, p, int(x1)), tm_array)


stop = timeit.default_timer()
print("Time", stop-start)

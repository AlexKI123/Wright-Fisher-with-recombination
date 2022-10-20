import numba as nb
import random
import numpy
from numba import prange
import math


@nb.njit
def ham_dist_fct_nb(a, b):
    x = a ^ b
    setBits = 0
    while x > 0:
        setBits += x & 1
        x >>= 1
    return setBits


def perc_model(c, p, viables, lethals):
    if c not in viables and c not in lethals:
        if random.random() <= p:
            viables.add(c)
        else:
            lethals.add(c)


@nb.njit
def mutational_robustness_nb_smr2_hd(N, wp, viables, l, two_point_distances):
    #wp_reduced = [0]
    #del wp_reduced[:]
    wp_reduced = []

    #wp_reduced_viable = [0]
    #del wp_reduced_viable[:]
    wp_reduced_viable = []

    for x3 in range(N):
        if wp[x3] not in wp_reduced:
            wp_reduced.append(wp[x3])
            if wp[x3] in viables:
                wp_reduced_viable.append(wp[x3])
    number_of_distinct_genotypes = len(wp_reduced)
    number_of_distinct_and_viable_genotypes = len(wp_reduced_viable)


    wp_counter = [0] * number_of_distinct_genotypes
    for x3 in range(number_of_distinct_genotypes):
        for x2 in range(N):
            if wp_reduced[x3] == wp[x2]:
                wp_counter[x3] += 1

    meanfitness = 0.0
    mutational_robustness_array = [0.0] * number_of_distinct_genotypes
    neighbors = [0]
    for individual in range(number_of_distinct_genotypes):
        if wp_reduced[individual] in viables:

            meanfitness += wp_counter[individual]

            #del neighbors[:]
            neighbors = []
            for loci in range(l):
                neighbors.append(wp_reduced[individual] ^ (1 << loci))
            for n in neighbors:
                if n in viables:
                    mutational_robustness_array[individual] += 1.0
            mutational_robustness_array[individual] = 1.0 * mutational_robustness_array[individual] / l
            mutational_robustness_array[individual] *= wp_counter[individual] / N
    meanfitness = 1.0*meanfitness/N
    mvsgen_at_current_gen = 0.0
    for x3 in range(number_of_distinct_genotypes):
        mvsgen_at_current_gen += 1.0 * mutational_robustness_array[x3]

    avg_population_hamdist = 0
    for x1 in range(len(wp_reduced) - 1):
        for x2 in range(x1 + 1, len(wp_reduced)):
            avg_population_hamdist += 1.0 * ham_dist_fct_nb(wp_reduced[x1], wp_reduced[x2]) * wp_counter[x1] * wp_counter[x2]
    avg_population_hamdist = avg_population_hamdist/two_point_distances


    # compute the number of segregating mutations
    and_, or_ = ~0, 0
    for x in wp_reduced:
        and_ &= x
        or_ |= x
    xor_ = and_ ^ or_
    numberofsegmut = 0
    while xor_ > 0:
        numberofsegmut += 1
        xor_ &= xor_ - 1

    return mvsgen_at_current_gen, avg_population_hamdist, number_of_distinct_genotypes, numberofsegmut, meanfitness, number_of_distinct_and_viable_genotypes


@nb.njit(parallel=True)
def sel_mut_rec(rmin, rmax, initial_generations, measure_generations, N, l, mu, datapoints, viables, two_point_distances, log):
    if log is False:
        rlist = numpy.linspace(rmin, rmax, datapoints)
    if log is True:
        rlist_temp = numpy.linspace(numpy.log10(rmin), numpy.log10(rmax), datapoints)
        rlist = numpy.power(10, rlist_temp)
    mr_r_hd_array = numpy.zeros((datapoints, 3))

    for x1 in prange(datapoints):

        wp = [0] * N
        wp_temp = [0]
        wp_temp2 = [0] * N
        for _ in range(initial_generations):
            # selection
            del wp_temp[:]
            while len(wp_temp) != N:
                individual = random.randint(0, N - 1)
                if wp[individual] in viables:
                    wp_temp.append(wp[individual])
            wp[:] = wp_temp

            # mutation
            for mutation_in_individual in range(N):
                for mutation_at_loci in range(l):
                    if random.random() < mu:
                        wp[mutation_in_individual] = wp[mutation_in_individual] ^ (1 << mutation_at_loci)

            # recombination
            for x3 in range(N):
                wp_temp2[x3] = wp[x3]
            for x3 in range(N):
                if random.random() <= rlist[x1]:
                    randint1 = random.randint(0, N - 1)
                    randint2 = random.randint(0, N - 1)
                    while randint1 == randint2:
                        randint1 = random.randint(0, N - 1)
                        randint2 = random.randint(0, N - 1)
                    a = wp[randint1]
                    b = wp[randint2]
                    c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                    wp_temp2[x3] = c
            for x3 in range(N):
                wp[x3] = wp_temp2[x3]

        m_value = 0.0
        avg_hd = 0.0
        for _ in range(measure_generations):
            # selection
            del wp_temp[:]
            while len(wp_temp) != N:
                individual = random.randint(0, N - 1)
                if wp[individual] in viables:
                    wp_temp.append(wp[individual])
            wp[:] = wp_temp

            # mutation
            for mutation_in_individual in range(N):
                for mutation_at_loci in range(l):
                    if random.random() < mu:
                        wp[mutation_in_individual] = wp[mutation_in_individual] ^ (1 << mutation_at_loci)

            # recombination
            for x3 in range(N):
                wp_temp2[x3] = wp[x3]
            for x3 in range(N):
                if random.random() <= rlist[x1]:
                    randint1 = random.randint(0, N - 1)
                    randint2 = random.randint(0, N - 1)
                    while randint1 == randint2:
                        randint1 = random.randint(0, N - 1)
                        randint2 = random.randint(0, N - 1)
                    a = wp[randint1]
                    b = wp[randint2]
                    c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                    wp_temp2[x3] = c
            for x3 in range(N):
                wp[x3] = wp_temp2[x3]

            temp_values = mutational_robustness_nb_smr2_hd(N, wp, viables, l, two_point_distances)
            m_value += 1.0 * temp_values[0]
            avg_hd += 1.0 * temp_values[1]

        mr_r_hd_array[x1, 0] = rlist[x1]
        mr_r_hd_array[x1, 1] = m_value / measure_generations
        mr_r_hd_array[x1, 2] = avg_hd / measure_generations

    return mr_r_hd_array


@nb.njit(parallel=True)
def sel_rec_mut(rmin, rmax, initial_generations, measure_generations, N, l, mu, datapoints, viables, two_point_distances, log):
    if log is False:
        rlist = numpy.linspace(rmin, rmax, datapoints)
    if log is True:
        rlist_temp = numpy.linspace(numpy.log10(rmin), numpy.log10(rmax), datapoints)
        rlist = numpy.power(10, rlist_temp)
    mr_r_hd_array = numpy.zeros((datapoints, 5))

    for x1 in prange(datapoints):

        wp = [0] * N
        wp_temp = [0]
        wp_temp2 = [0] * N
        NL = N * l  # mutation v2 & v3
        NL_range_tempalte = list(range(NL))  # mutation v3
        selected = {0, 1, 2}  # mutation v3

        for _ in range(initial_generations):
            segMut_temp = len(set(wp))
            if segMut_temp > 1:
                # selection
                del wp_temp[:]
                while len(wp_temp) != N:
                    individual = random.randint(0, N - 1)
                    if wp[individual] in viables:
                        wp_temp.append(wp[individual])
                wp[:] = wp_temp

                # recombination (uniform)
                for x3 in range(N):
                    wp_temp2[x3] = wp[x3]
                for x3 in range(N):
                    if random.random() <= rlist[x1]:
                        randint1 = random.randint(0, N - 1)
                        randint2 = random.randint(0, N - 1)
                        while randint1 == randint2:
                            randint1 = random.randint(0, N - 1)
                            randint2 = random.randint(0, N - 1)
                        a = wp[randint1]
                        b = wp[randint2]
                        c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                        wp_temp2[x3] = c
                for x3 in range(N):
                    wp[x3] = wp_temp2[x3]

            # mutation v3
            flips = numpy.random.binomial(NL, mu)
            if flips > 0:
                positions = [0] * flips
                setsize = 21  # size of a small set minus size of an empty list
                if flips > 5:
                    setsize += 4 ** math.ceil(math.log(flips * 3) / math.log(4))  # table size for big sets
                if NL <= setsize:
                    # An n-length list is smaller than a k-length set
                    pool = NL_range_tempalte[:]
                    for i in range(flips):  # invariant:  non-selected at [0,n-i)
                        j = random.randint(0, (NL - i) - 1)
                        positions[i] = pool[j]
                        pool[j] = pool[NL - i - 1]  # move non-selected item into vacancy
                else:
                    # selected = set()
                    selected.clear()
                    selected_add = selected.add
                    for i in range(flips):
                        j = random.randint(0, NL - 1)
                        while j in selected:
                            j = random.randint(0, NL - 1)
                        selected_add(j)
                        positions[i] = NL_range_tempalte[j]

                for pos in positions:
                    wp[pos // l] = wp[pos // l] ^ (1 << (pos % l))

        m_value = 0.0
        avg_hd = 0.0
        nogenotypes = 0.0
        nosegmut = 0.0
        for _ in range(measure_generations):
            segMut_temp = len(set(wp))
            # if not [wp[0]] * len(wp) == wp:
            if segMut_temp > 1:
                # selection
                del wp_temp[:]
                while len(wp_temp) != N:
                    individual = random.randint(0, N - 1)
                    if wp[individual] in viables:
                        wp_temp.append(wp[individual])
                wp[:] = wp_temp

                # recombination (uniform)
                for x3 in range(N):
                    wp_temp2[x3] = wp[x3]
                for x3 in range(N):
                    if random.random() <= rlist[x1]:
                        randint1 = random.randint(0, N - 1)
                        randint2 = random.randint(0, N - 1)
                        while randint1 == randint2:
                            randint1 = random.randint(0, N - 1)
                            randint2 = random.randint(0, N - 1)
                        a = wp[randint1]
                        b = wp[randint2]
                        c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                        wp_temp2[x3] = c
                for x3 in range(N):
                    wp[x3] = wp_temp2[x3]

            # mutation v3
            flips = numpy.random.binomial(NL, mu)
            if flips > 0:
                positions = [0] * flips
                setsize = 21  # size of a small set minus size of an empty list
                if flips > 5:
                    setsize += 4 ** math.ceil(math.log(flips * 3) / math.log(4))  # table size for big sets
                if NL <= setsize:
                    # An n-length list is smaller than a k-length set
                    pool = NL_range_tempalte[:]
                    for i in range(flips):  # invariant:  non-selected at [0,n-i)
                        j = random.randint(0, (NL - i) - 1)
                        positions[i] = pool[j]
                        pool[j] = pool[NL - i - 1]  # move non-selected item into vacancy
                else:
                    # selected = set()
                    selected.clear()
                    selected_add = selected.add
                    for i in range(flips):
                        j = random.randint(0, NL - 1)
                        while j in selected:
                            j = random.randint(0, NL - 1)
                        selected_add(j)
                        positions[i] = NL_range_tempalte[j]

                for pos in positions:
                    wp[pos // l] = wp[pos // l] ^ (1 << (pos % l))


            if flips > 0 or segMut_temp > 1 or _ == 0:
                temp_values = mutational_robustness_nb_smr2_hd(N, wp, viables, l, two_point_distances)
            m_value += 1.0 * temp_values[0]
            avg_hd += 1.0 * temp_values[1]
            nogenotypes += 1.0 * temp_values[2]
            nosegmut += 1.0 * temp_values[3]

        mr_r_hd_array[x1, 0] = rlist[x1]
        mr_r_hd_array[x1, 1] = m_value / measure_generations
        mr_r_hd_array[x1, 2] = avg_hd / measure_generations
        mr_r_hd_array[x1, 3] = nogenotypes / measure_generations
        mr_r_hd_array[x1, 4] = nosegmut / measure_generations

    return mr_r_hd_array


@nb.njit(parallel=True)
def sel_rec_mut_ffpopsim(rmin, rmax, initial_generations, measure_generations, N, l, mu, datapoints, viables, two_point_distances, log):
    if log is False:
        rlist = numpy.linspace(rmin, rmax, datapoints)
    if log is True:
        rlist_temp = numpy.linspace(numpy.log10(rmin), numpy.log10(rmax), datapoints)
        rlist = numpy.power(10, rlist_temp)
    mr_r_hd_array = numpy.zeros((datapoints, 5))

    for x1 in prange(datapoints):

        wp = [0] * N
        wp_temp = [0]
        wp_temp2 = [0] * N
        NL = N * l  # mutation v2 & v3
        NL_range_tempalte = list(range(NL))  # mutation v3
        selected = {0, 1, 2}  # mutation v3
        recombining_ind = [0]

        for _ in range(initial_generations):
            segMut_temp = len(set(wp))
            if segMut_temp > 1:
                # selection
                del wp_temp[:]
                while len(wp_temp) != N:
                    individual = random.randint(0, N - 1)
                    if wp[individual] in viables:
                        wp_temp.append(wp[individual])
                wp[:] = wp_temp

                # recombination
                del recombining_ind[:]
                for x3 in range(N):
                    if random.random() <= rlist[x1]:
                        recombining_ind.append(x3)
                count_recombining_ind = len(recombining_ind)
                if count_recombining_ind > 1:
                    if count_recombining_ind % 2 != 0:
                        recombining_ind.pop(random.randint(0, count_recombining_ind-1))
                    count_recombining_ind = len(recombining_ind)
                    count_recombining_ind = int(count_recombining_ind/2)
                    for x3 in range(N):
                        wp_temp2[x3] = wp[x3]
                    for x3 in range(count_recombining_ind):
                        randint1 = random.randint(0, len(recombining_ind) - 1)
                        randind1 = recombining_ind[randint1]
                        recombining_ind.pop(randint1)
                        a = wp[randind1]

                        randint2 = random.randint(0, len(recombining_ind) - 1)
                        randind2 = recombining_ind[randint2]
                        recombining_ind.pop(randint2)
                        b = wp[randind2]

                        if randind1 == randind2:
                            print("ERROR")

                        c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                        wp_temp2[randind1] = c
                        #c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                        c = (a ^ b) ^ c
                        wp_temp2[randind2] = c
                    for x3 in range(N):
                        wp[x3] = wp_temp2[x3]

            # mutation v3
            flips = numpy.random.binomial(NL, mu)
            if flips > 0:
                positions = [0] * flips
                setsize = 21  # size of a small set minus size of an empty list
                if flips > 5:
                    setsize += 4 ** math.ceil(math.log(flips * 3) / math.log(4))  # table size for big sets
                if NL <= setsize:
                    # An n-length list is smaller than a k-length set
                    pool = NL_range_tempalte[:]
                    for i in range(flips):  # invariant:  non-selected at [0,n-i)
                        j = random.randint(0, (NL - i) - 1)
                        positions[i] = pool[j]
                        pool[j] = pool[NL - i - 1]  # move non-selected item into vacancy
                else:
                    # selected = set()
                    selected.clear()
                    selected_add = selected.add
                    for i in range(flips):
                        j = random.randint(0, NL - 1)
                        while j in selected:
                            j = random.randint(0, NL - 1)
                        selected_add(j)
                        positions[i] = NL_range_tempalte[j]

                for pos in positions:
                    wp[pos // l] = wp[pos // l] ^ (1 << (pos % l))


        m_value = 0.0
        avg_hd = 0.0
        nogenotypes = 0.0
        nosegmut = 0.0
        for _ in range(measure_generations):
            segMut_temp = len(set(wp))
            # if not [wp[0]] * len(wp) == wp:
            if segMut_temp > 1:
                # selection
                del wp_temp[:]
                while len(wp_temp) != N:
                    individual = random.randint(0, N - 1)
                    if wp[individual] in viables:
                        wp_temp.append(wp[individual])
                wp[:] = wp_temp

                # recombination
                del recombining_ind[:]
                for x3 in range(N):
                    if random.random() <= rlist[x1]:
                        recombining_ind.append(x3)
                count_recombining_ind = len(recombining_ind)
                if count_recombining_ind > 1:
                    if count_recombining_ind % 2 != 0:
                        recombining_ind.pop(random.randint(0, count_recombining_ind - 1))
                    count_recombining_ind = len(recombining_ind)
                    count_recombining_ind = int(count_recombining_ind / 2)
                    for x3 in range(N):
                        wp_temp2[x3] = wp[x3]
                    for x3 in range(count_recombining_ind):
                        randint1 = random.randint(0, len(recombining_ind) - 1)
                        randind1 = recombining_ind[randint1]
                        recombining_ind.pop(randint1)
                        a = wp[randind1]

                        randint2 = random.randint(0, len(recombining_ind) - 1)
                        randind2 = recombining_ind[randint2]
                        recombining_ind.pop(randint2)
                        b = wp[randind2]

                        c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                        wp_temp2[randind1] = c
                        # c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                        c = (a ^ b) ^ c
                        wp_temp2[randind2] = c
                    for x3 in range(N):
                        wp[x3] = wp_temp2[x3]


            # mutation v3
            flips = numpy.random.binomial(NL, mu)
            if flips > 0:
                positions = [0] * flips
                setsize = 21  # size of a small set minus size of an empty list
                if flips > 5:
                    setsize += 4 ** math.ceil(math.log(flips * 3) / math.log(4))  # table size for big sets
                if NL <= setsize:
                    # An n-length list is smaller than a k-length set
                    pool = NL_range_tempalte[:]
                    for i in range(flips):  # invariant:  non-selected at [0,n-i)
                        j = random.randint(0, (NL - i) - 1)
                        positions[i] = pool[j]
                        pool[j] = pool[NL - i - 1]  # move non-selected item into vacancy
                else:
                    # selected = set()
                    selected.clear()
                    selected_add = selected.add
                    for i in range(flips):
                        j = random.randint(0, NL - 1)
                        while j in selected:
                            j = random.randint(0, NL - 1)
                        selected_add(j)
                        positions[i] = NL_range_tempalte[j]

                for pos in positions:
                    wp[pos // l] = wp[pos // l] ^ (1 << (pos % l))


            if flips > 0 or segMut_temp > 1 or _ == 0:
                temp_values = mutational_robustness_nb_smr2_hd(N, wp, viables, l, two_point_distances)
            m_value += 1.0 * temp_values[0]
            avg_hd += 1.0 * temp_values[1]
            nogenotypes += 1.0 * temp_values[2]
            nosegmut += 1.0 * temp_values[3]

        mr_r_hd_array[x1, 0] = rlist[x1]
        mr_r_hd_array[x1, 1] = m_value / measure_generations
        mr_r_hd_array[x1, 2] = avg_hd / measure_generations
        mr_r_hd_array[x1, 3] = nogenotypes / measure_generations
        mr_r_hd_array[x1, 4] = nosegmut / measure_generations

    return mr_r_hd_array


@nb.njit(parallel=True)
def sel_rec_mut_opc(rmin, rmax, initial_generations, measure_generations, N, l, mu, datapoints, viables, two_point_distances, log):
    if log is False:
        rlist = numpy.linspace(rmin, rmax, datapoints)
    if log is True:
        rlist_temp = numpy.linspace(numpy.log10(rmin), numpy.log10(rmax), datapoints)
        rlist = numpy.power(10, rlist_temp)
    mr_r_hd_array = numpy.zeros((datapoints, 5))

    for x1 in prange(datapoints):

        wp = [0] * N
        wp_temp = [0]
        wp_temp2 = [0] * N
        NL = N * l  # mutation v2 & v3
        NL_range_tempalte = list(range(NL))  # mutation v3
        selected = {0, 1, 2}  # mutation v3

        for _ in range(initial_generations):
            segMut_temp = len(set(wp))
            if segMut_temp > 1:
                # selection
                del wp_temp[:]
                while len(wp_temp) != N:
                    individual = random.randint(0, N - 1)
                    if wp[individual] in viables:
                        wp_temp.append(wp[individual])
                wp[:] = wp_temp

                # recombination (one-point)
                for x3 in range(N):
                    wp_temp2[x3] = wp[x3]
                for x3 in range(N):
                    if random.random() <= rlist[x1]:
                        randint1 = random.randint(0, N - 1)
                        randint2 = random.randint(0, N - 1)
                        while randint1 == randint2:
                            randint1 = random.randint(0, N - 1)
                        a = wp[randint1]
                        b = wp[randint2]
                        randompos = numpy.random.randint(1, l)
                        coinflip = numpy.random.randint(2)
                        m = (1 << randompos) - 1
                        c = (a if coinflip else b) ^ ((a ^ b) & m)
                        wp_temp2[x3] = c
                for x3 in range(N):
                    wp[x3] = wp_temp2[x3]

            # mutation v3
            flips = numpy.random.binomial(NL, mu)
            if flips > 0:
                positions = [0] * flips
                setsize = 21  # size of a small set minus size of an empty list
                if flips > 5:
                    setsize += 4 ** math.ceil(math.log(flips * 3) / math.log(4))  # table size for big sets
                if NL <= setsize:
                    # An n-length list is smaller than a k-length set
                    pool = NL_range_tempalte[:]
                    for i in range(flips):  # invariant:  non-selected at [0,n-i)
                        j = random.randint(0, (NL - i) - 1)
                        positions[i] = pool[j]
                        pool[j] = pool[NL - i - 1]  # move non-selected item into vacancy
                else:
                    # selected = set()
                    selected.clear()
                    selected_add = selected.add
                    for i in range(flips):
                        j = random.randint(0, NL - 1)
                        while j in selected:
                            j = random.randint(0, NL - 1)
                        selected_add(j)
                        positions[i] = NL_range_tempalte[j]

                for pos in positions:
                    wp[pos // l] = wp[pos // l] ^ (1 << (pos % l))


        m_value = 0.0
        avg_hd = 0.0
        nogenotypes = 0.0
        nosegmut = 0.0
        for _ in range(measure_generations):
            segMut_temp = len(set(wp))
            # if not [wp[0]] * len(wp) == wp:
            if segMut_temp > 1:
                # selection
                del wp_temp[:]
                while len(wp_temp) != N:
                    individual = random.randint(0, N - 1)
                    if wp[individual] in viables:
                        wp_temp.append(wp[individual])
                wp[:] = wp_temp


                # recombination (one-point)
                for x3 in range(N):
                    wp_temp2[x3] = wp[x3]
                for x3 in range(N):
                    if random.random() <= rlist[x1]:
                        randint1 = random.randint(0, N - 1)
                        randint2 = random.randint(0, N - 1)
                        while randint1 == randint2:
                            randint1 = random.randint(0, N - 1)
                        a = wp[randint1]
                        b = wp[randint2]
                        randompos = numpy.random.randint(1, l)
                        coinflip = numpy.random.randint(2)
                        m = (1 << randompos) - 1
                        c = (a if coinflip else b) ^ ((a ^ b) & m)
                        wp_temp2[x3] = c
                for x3 in range(N):
                    wp[x3] = wp_temp2[x3]

            # mutation v3
            flips = numpy.random.binomial(NL, mu)
            if flips > 0:
                positions = [0] * flips
                setsize = 21  # size of a small set minus size of an empty list
                if flips > 5:
                    setsize += 4 ** math.ceil(math.log(flips * 3) / math.log(4))  # table size for big sets
                if NL <= setsize:
                    # An n-length list is smaller than a k-length set
                    pool = NL_range_tempalte[:]
                    for i in range(flips):  # invariant:  non-selected at [0,n-i)
                        j = random.randint(0, (NL - i) - 1)
                        positions[i] = pool[j]
                        pool[j] = pool[NL - i - 1]  # move non-selected item into vacancy
                else:
                    # selected = set()
                    selected.clear()
                    selected_add = selected.add
                    for i in range(flips):
                        j = random.randint(0, NL - 1)
                        while j in selected:
                            j = random.randint(0, NL - 1)
                        selected_add(j)
                        positions[i] = NL_range_tempalte[j]

                for pos in positions:
                    wp[pos // l] = wp[pos // l] ^ (1 << (pos % l))


            if flips > 0 or segMut_temp > 1 or _ == 0:
                temp_values = mutational_robustness_nb_smr2_hd(N, wp, viables, l, two_point_distances)
            m_value += 1.0 * temp_values[0]
            avg_hd += 1.0 * temp_values[1]
            nogenotypes += 1.0 * temp_values[2]
            nosegmut += 1.0 * temp_values[3]


        mr_r_hd_array[x1, 0] = rlist[x1]
        mr_r_hd_array[x1, 1] = m_value / measure_generations
        mr_r_hd_array[x1, 2] = avg_hd / measure_generations
        mr_r_hd_array[x1, 3] = nogenotypes / measure_generations
        mr_r_hd_array[x1, 4] = nosegmut / measure_generations

    return mr_r_hd_array


@nb.njit(parallel=True)
def selrec_mut(rmin, rmax, initial_generations, measure_generations, N, l, mu, datapoints, viables, lethals, two_point_distances, log):

    if log is False:
        rlist = numpy.linspace(rmin, rmax, datapoints)
    if log is True:
        rlist_temp = numpy.linspace(numpy.log10(rmin), numpy.log10(rmax), datapoints)
        rlist = numpy.power(10, rlist_temp)


    mr_r_hd_array = numpy.zeros((datapoints, 5))

    for x1 in prange(datapoints):

        wp = [0] * N
        wp_temp2 = [0] * N
        for _ in range(initial_generations):

            # selection + recombination
            for x3 in range(N):
                wp_temp2[x3] = wp[x3]
            for x3 in range(N):
                if random.random() <= rlist[x1]:
                    randint1 = random.randint(0, N - 1)
                    randint2 = random.randint(0, N - 1)
                    while randint1 == randint2 or wp[randint1] in lethals or wp[randint2] in lethals:
                        randint1 = random.randint(0, N - 1)
                        randint2 = random.randint(0, N - 1)
                    a = wp[randint1]
                    b = wp[randint2]
                    c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                    wp_temp2[x3] = c
                else:
                    randint1 = random.randint(0, N - 1)
                    while wp[randint1] in lethals:
                        randint1 = random.randint(0, N - 1)
                    c = wp[randint1]
                    wp_temp2[x3] = c
            for x3 in range(N):
                wp[x3] = wp_temp2[x3]

            # mutation
            for mutation_in_individual in range(N):
                for mutation_at_loci in range(l):
                    if random.random() < mu:
                        wp[mutation_in_individual] = wp[mutation_in_individual] ^ (1 << mutation_at_loci)

        m_value = 0.0
        avg_hd = 0.0
        nogenotypes = 0.0
        nosegmut = 0.0
        for _ in range(measure_generations):

            # selection + recombination
            for x3 in range(N):
                wp_temp2[x3] = wp[x3]
            for x3 in range(N):
                if random.random() <= rlist[x1]:
                    randint1 = random.randint(0, N - 1)
                    randint2 = random.randint(0, N - 1)
                    while randint1 == randint2 or wp[randint1] in lethals or wp[randint2] in lethals:
                        randint1 = random.randint(0, N - 1)
                        randint2 = random.randint(0, N - 1)
                    a = wp[randint1]
                    b = wp[randint2]
                    c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                    wp_temp2[x3] = c
                else:
                    randint1 = random.randint(0, N - 1)
                    while wp[randint1] in lethals:
                        randint1 = random.randint(0, N - 1)
                    c = wp[randint1]
                    wp_temp2[x3] = c
            for x3 in range(N):
                wp[x3] = wp_temp2[x3]

            # mutation
            for mutation_in_individual in range(N):
                for mutation_at_loci in range(l):
                    if random.random() < mu:
                        wp[mutation_in_individual] = wp[mutation_in_individual] ^ (1 << mutation_at_loci)

            temp_values = mutational_robustness_nb_smr2_hd(N, wp, viables, l, two_point_distances)
            m_value += 1.0 * temp_values[0]
            avg_hd += 1.0 * temp_values[1]
            nogenotypes += 1.0 * temp_values[2]
            nosegmut += 1.0 * temp_values[3]

        mr_r_hd_array[x1, 0] = rlist[x1]
        mr_r_hd_array[x1, 1] = m_value / measure_generations
        mr_r_hd_array[x1, 2] = avg_hd / measure_generations
        mr_r_hd_array[x1, 3] = nogenotypes / measure_generations
        mr_r_hd_array[x1, 4] = nosegmut / measure_generations

    return mr_r_hd_array


@nb.njit(parallel=True)
def selrec_mut_onlyviables(rmin, rmax, initial_generations, measure_generations, N, l, mu, datapoints, viables, two_point_distances, log):

    if log is False:
        rlist = numpy.linspace(rmin, rmax, datapoints)
    if log is True:
        rlist_temp = numpy.linspace(numpy.log10(rmin), numpy.log10(rmax), datapoints)
        rlist = numpy.power(10, rlist_temp)

    mr_r_hd_array = numpy.zeros((datapoints, 5))

    for x1 in prange(datapoints):

        wp = [0] * N
        wp_temp2 = [0] * N
        for _ in range(initial_generations):

            # selection + recombination
            for x3 in range(N):
                wp_temp2[x3] = wp[x3]
            for x3 in range(N):
                if random.random() <= rlist[x1]:
                    randint1 = random.randint(0, N - 1)
                    randint2 = random.randint(0, N - 1)
                    while randint1 == randint2:
                        randint1 = random.randint(0, N - 1)
                        randint2 = random.randint(0, N - 1)
                    a = wp[randint1]
                    b = wp[randint2]
                    c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                    wp_temp2[x3] = c
                else:
                    randint1 = random.randint(0, N - 1)
                    c = wp[randint1]
                    wp_temp2[x3] = c
            for x3 in range(N):
                wp[x3] = wp_temp2[x3]

            # mutation
            for mutation_in_individual in range(N):
                for mutation_at_loci in range(l):
                    if random.random() < mu:
                        wp[mutation_in_individual] = wp[mutation_in_individual] ^ (1 << mutation_at_loci)

        m_value = 0.0
        avg_hd = 0.0
        nogenotypes = 0.0
        nosegmut = 0.0
        for _ in range(measure_generations):

            # selection + recombination
            for x3 in range(N):
                wp_temp2[x3] = wp[x3]
            for x3 in range(N):
                if random.random() <= rlist[x1]:
                    randint1 = random.randint(0, N - 1)
                    randint2 = random.randint(0, N - 1)
                    while randint1 == randint2:
                        randint1 = random.randint(0, N - 1)
                        randint2 = random.randint(0, N - 1)
                    a = wp[randint1]
                    b = wp[randint2]
                    c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                    wp_temp2[x3] = c
                else:
                    randint1 = random.randint(0, N - 1)
                    c = wp[randint1]
                    wp_temp2[x3] = c
            for x3 in range(N):
                wp[x3] = wp_temp2[x3]

            # mutation
            for mutation_in_individual in range(N):
                for mutation_at_loci in range(l):
                    if random.random() < mu:
                        wp[mutation_in_individual] = wp[mutation_in_individual] ^ (1 << mutation_at_loci)

            temp_values = mutational_robustness_nb_smr2_hd(N, wp, viables, l, two_point_distances)
            m_value += 1.0 * temp_values[0]
            avg_hd += 1.0 * temp_values[1]
            nogenotypes += 1.0 * temp_values[2]
            nosegmut += 1.0 * temp_values[3]

        mr_r_hd_array[x1, 0] = rlist[x1]
        mr_r_hd_array[x1, 1] = m_value / measure_generations
        mr_r_hd_array[x1, 2] = avg_hd / measure_generations
        mr_r_hd_array[x1, 3] = nogenotypes / measure_generations
        mr_r_hd_array[x1, 4] = nosegmut / measure_generations

    return mr_r_hd_array


@nb.njit()
def sel_mut_rec_cluster(r, initial_generations, measure_generations, N, l, mu, viables, two_point_distances):
    wp = [0] * N
    wp_temp = [0]
    wp_temp2 = [0] * N
    for _ in range(initial_generations):
        #selection
        del wp_temp[:]
        while len(wp_temp) != N:
            individual = random.randint(0, N - 1)
            if wp[individual] in viables:
                wp_temp.append(wp[individual])
        wp[:] = wp_temp

        #mutation
        for mutation_in_individual in range(N):
            for mutation_at_loci in range(l):
                if random.random() < mu:
                    wp[mutation_in_individual] = wp[mutation_in_individual] ^ (1 << mutation_at_loci)

        #recombination
        for x3 in range(N):
            wp_temp2[x3] = wp[x3]
        for x3 in range(N):
            if random.random() <= r:
                randint1 = random.randint(0, N - 1)
                randint2 = random.randint(0, N - 1)
                while randint1 == randint2:
                    randint1 = random.randint(0, N - 1)
                    randint2 = random.randint(0, N - 1)
                a = wp[randint1]
                b = wp[randint2]
                c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                wp_temp2[x3] = c
        for x3 in range(N):
            wp[x3] = wp_temp2[x3]


    m_value = 0.0
    avg_hd = 0.0
    nogenotypes = 0.0
    nosegmut = 0.0
    for _ in range(measure_generations):
        #selection
        del wp_temp[:]
        while len(wp_temp) != N:
            individual = random.randint(0, N - 1)
            if wp[individual] in viables:
                wp_temp.append(wp[individual])
        wp[:] = wp_temp

        #mutation
        for mutation_in_individual in range(N):
            for mutation_at_loci in range(l):
                if random.random() < mu:
                    wp[mutation_in_individual] = wp[mutation_in_individual] ^ (1 << mutation_at_loci)

        #recombination
        for x3 in range(N):
            wp_temp2[x3] = wp[x3]
        for x3 in range(N):
            if random.random() <= r:
                randint1 = random.randint(0, N - 1)
                randint2 = random.randint(0, N - 1)
                while randint1 == randint2:
                    randint1 = random.randint(0, N - 1)
                    randint2 = random.randint(0, N - 1)
                a = wp[randint1]
                b = wp[randint2]
                c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                wp_temp2[x3] = c
        for x3 in range(N):
            wp[x3] = wp_temp2[x3]

        temp_values = mutational_robustness_nb_smr2_hd(N, wp, viables, l, two_point_distances)
        m_value += 1.0 * temp_values[0]
        avg_hd += 1.0 * temp_values[1]
        nogenotypes += 1.0 * temp_values[2]
        nosegmut += 1.0 * temp_values[3]

    m_value = m_value / measure_generations
    avg_hd = avg_hd / measure_generations
    nogenotypes = nogenotypes / measure_generations
    nosegmut = nosegmut / measure_generations

    return m_value, avg_hd, nogenotypes, nosegmut


@nb.njit()
def sel_rec_mut_cluster(r, initial_generations, measure_generations, N, l, mu, viables, two_point_distances):
    wp = [0] * N
    wp_temp = [0]
    wp_temp2 = [0] * N
    NL = N * l  # mutation v2 & v3
    #positions = [0]  # mutation v2
    NL_range_tempalte = list(range(NL))  # mutation v3
    selected = {0, 1, 2} #mutation v3

    for _ in range(initial_generations):

        segMut_temp = len(set(wp))
        # if not [wp[0]] * len(wp) == wp:
        if segMut_temp > 1:
            #selection
            del wp_temp[:]
            while len(wp_temp) != N:
                individual = random.randint(0, N - 1)
                if wp[individual] in viables:
                    wp_temp.append(wp[individual])
            wp[:] = wp_temp

            #recombination (uniform)
            for x3 in range(N):
                wp_temp2[x3] = wp[x3]
            for x3 in range(N):
                if random.random() <= r:
                    randint1 = random.randint(0, N - 1)
                    randint2 = random.randint(0, N - 1)
                    while randint1 == randint2:
                        randint1 = random.randint(0, N - 1)
                        randint2 = random.randint(0, N - 1)
                    a = wp[randint1]
                    b = wp[randint2]
                    c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                    wp_temp2[x3] = c
            for x3 in range(N):
                wp[x3] = wp_temp2[x3]

        """
        #mutation
        for mutation_in_individual in range(N):
            for mutation_at_loci in range(l):
                if random.random() < mu:
                    wp[mutation_in_individual] = wp[mutation_in_individual] ^ (1 << mutation_at_loci)
        """

        """
         #mutation v2
         flips = numpy.random.binomial(NL, mu)
         if flips > 0:
             NL_range = list(range(NL))
             del positions[:]
             for x in range(flips):
                 randint = random.randint(0, len(NL_range) - 1)
                 positions.append(NL_range.pop(randint))
             for pos in positions:
                 wp[pos // l] = wp[pos // l] ^ (1 << (pos % l))
         """

        # mutation v3
        flips = numpy.random.binomial(NL, mu)
        if flips > 0:
            positions = [0] * flips
            setsize = 21  # size of a small set minus size of an empty list
            if flips > 5:
                setsize += 4 ** math.ceil(math.log(flips * 3) / math.log(4))  # table size for big sets
            if NL <= setsize:
                # An n-length list is smaller than a k-length set
                pool = NL_range_tempalte[:]
                for i in range(flips):  # invariant:  non-selected at [0,n-i)
                    j = random.randint(0, (NL - i) - 1)
                    positions[i] = pool[j]
                    pool[j] = pool[NL - i - 1]  # move non-selected item into vacancy
            else:
                # selected = set()
                selected.clear()
                selected_add = selected.add
                for i in range(flips):
                    j = random.randint(0, NL - 1)
                    while j in selected:
                        j = random.randint(0, NL - 1)
                    selected_add(j)
                    positions[i] = NL_range_tempalte[j]

            for pos in positions:
                wp[pos // l] = wp[pos // l] ^ (1 << (pos % l))

    m_value = 0.0
    avg_hd = 0.0
    nogenotypes = 0.0
    nosegmut = 0.0
    for _ in range(measure_generations):

        segMut_temp = len(set(wp))
        # if not [wp[0]] * len(wp) == wp:
        if segMut_temp > 1:
            #selection
            del wp_temp[:]
            while len(wp_temp) != N:
                individual = random.randint(0, N - 1)
                if wp[individual] in viables:
                    wp_temp.append(wp[individual])
            wp[:] = wp_temp

            # recombination (uniform)
            for x3 in range(N):
                wp_temp2[x3] = wp[x3]
            for x3 in range(N):
                if random.random() <= r:
                    randint1 = random.randint(0, N - 1)
                    randint2 = random.randint(0, N - 1)
                    while randint1 == randint2:
                        randint1 = random.randint(0, N - 1)
                        randint2 = random.randint(0, N - 1)
                    a = wp[randint1]
                    b = wp[randint2]
                    c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                    wp_temp2[x3] = c
            for x3 in range(N):
                wp[x3] = wp_temp2[x3]


        """
        #mutation
        for mutation_in_individual in range(N):
            for mutation_at_loci in range(l):
                if random.random() < mu:
                    wp[mutation_in_individual] = wp[mutation_in_individual] ^ (1 << mutation_at_loci)
        """

        """
           #mutation v2
           flips = numpy.random.binomial(NL, mu)
           if flips > 0:
               NL_range = list(range(NL))
               del positions[:]
               for x in range(flips):
                   randint = random.randint(0, len(NL_range) - 1)
                   positions.append(NL_range.pop(randint))
               for pos in positions:
                   wp[pos // l] = wp[pos // l] ^ (1 << (pos % l))
           """

        # mutation v3
        flips = numpy.random.binomial(NL, mu)
        if flips > 0:
            positions = [0] * flips
            setsize = 21  # size of a small set minus size of an empty list
            if flips > 5:
                setsize += 4 ** math.ceil(math.log(flips * 3) / math.log(4))  # table size for big sets
            if NL <= setsize:
                # An n-length list is smaller than a k-length set
                pool = NL_range_tempalte[:]
                for i in range(flips):  # invariant:  non-selected at [0,n-i)
                    j = random.randint(0, (NL - i) - 1)
                    positions[i] = pool[j]
                    pool[j] = pool[NL - i - 1]  # move non-selected item into vacancy
            else:
                # selected = set()
                selected.clear()
                selected_add = selected.add
                for i in range(flips):
                    j = random.randint(0, NL - 1)
                    while j in selected:
                        j = random.randint(0, NL - 1)
                    selected_add(j)
                    positions[i] = NL_range_tempalte[j]

            for pos in positions:
                wp[pos // l] = wp[pos // l] ^ (1 << (pos % l))

        if flips > 0 or segMut_temp > 1 or _ == 0:
            temp_values = mutational_robustness_nb_smr2_hd(N, wp, viables, l, two_point_distances)
        m_value += 1.0 * temp_values[0]
        avg_hd += 1.0 * temp_values[1]
        nogenotypes += 1.0 * temp_values[2]
        nosegmut += 1.0 * temp_values[3]

    m_value = m_value / measure_generations
    avg_hd = avg_hd / measure_generations
    nogenotypes = nogenotypes / measure_generations
    nosegmut = nosegmut / measure_generations

    return m_value, avg_hd, nogenotypes, nosegmut


@nb.njit()
def selrec_mut_cluster(r, initial_generations, measure_generations, N, l, mu, viables, lethals, two_point_distances):
    wp = [0] * N
    wp_temp2 = [0] * N
    NL = N * l  # mutation v2 & v3
    #positions = [0]  # mutation v2
    NL_range_tempalte = list(range(NL))  # mutation v3
    selected = {0, 1, 2} #mutation v3

    for _ in range(initial_generations):

        segMut_temp = len(set(wp))
        #if not [wp[0]] * len(wp) == wp:
        if segMut_temp > 1:
            # selection + recombination
            for x3 in range(N):
                wp_temp2[x3] = wp[x3]
            for x3 in range(N):
                if random.random() <= r:
                    randint1 = random.randint(0, N - 1)
                    randint2 = random.randint(0, N - 1)
                    while randint1 == randint2 or wp[randint1] in lethals or wp[randint2] in lethals:
                        randint1 = random.randint(0, N - 1)
                        randint2 = random.randint(0, N - 1)
                    a = wp[randint1]
                    b = wp[randint2]
                    c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                    wp_temp2[x3] = c
                else:
                    randint1 = random.randint(0, N - 1)
                    while wp[randint1] in lethals:
                        randint1 = random.randint(0, N - 1)
                    c = wp[randint1]
                    wp_temp2[x3] = c
            for x3 in range(N):
                wp[x3] = wp_temp2[x3]

        # mutation v3
        flips = numpy.random.binomial(NL, mu)
        if flips > 0:
            positions = [0] * flips
            setsize = 21  # size of a small set minus size of an empty list
            if flips > 5:
                setsize += 4 ** math.ceil(math.log(flips * 3) / math.log(4))  # table size for big sets
            if NL <= setsize:
                # An n-length list is smaller than a k-length set
                pool = NL_range_tempalte[:]
                for i in range(flips):  # invariant:  non-selected at [0,n-i)
                    j = random.randint(0, (NL - i) - 1)
                    positions[i] = pool[j]
                    pool[j] = pool[NL - i - 1]  # move non-selected item into vacancy
            else:
                # selected = set()
                selected.clear()
                selected_add = selected.add
                for i in range(flips):
                    j = random.randint(0, NL - 1)
                    while j in selected:
                        j = random.randint(0, NL - 1)
                    selected_add(j)
                    positions[i] = NL_range_tempalte[j]

            for pos in positions:
                wp[pos // l] = wp[pos // l] ^ (1 << (pos % l))

    m_value = 0.0
    avg_hd = 0.0
    nogenotypes = 0.0
    nosegmut = 0.0
    for _ in range(measure_generations):

        segMut_temp = len(set(wp))
        # if not [wp[0]] * len(wp) == wp:
        if segMut_temp > 1:
            # selection + recombination
            for x3 in range(N):
                wp_temp2[x3] = wp[x3]
            for x3 in range(N):
                if random.random() <= r:
                    randint1 = random.randint(0, N - 1)
                    randint2 = random.randint(0, N - 1)
                    while randint1 == randint2 or wp[randint1] in lethals or wp[randint2] in lethals:
                        randint1 = random.randint(0, N - 1)
                        randint2 = random.randint(0, N - 1)
                    a = wp[randint1]
                    b = wp[randint2]
                    c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                    wp_temp2[x3] = c
                else:
                    randint1 = random.randint(0, N - 1)
                    while wp[randint1] in lethals:
                        randint1 = random.randint(0, N - 1)
                    c = wp[randint1]
                    wp_temp2[x3] = c
            for x3 in range(N):
                wp[x3] = wp_temp2[x3]

        # mutation v3
        flips = numpy.random.binomial(NL, mu)
        if flips > 0:
            positions = [0] * flips
            setsize = 21  # size of a small set minus size of an empty list
            if flips > 5:
                setsize += 4 ** math.ceil(math.log(flips * 3) / math.log(4))  # table size for big sets
            if NL <= setsize:
                # An n-length list is smaller than a k-length set
                pool = NL_range_tempalte[:]
                for i in range(flips):  # invariant:  non-selected at [0,n-i)
                    j = random.randint(0, (NL - i) - 1)
                    positions[i] = pool[j]
                    pool[j] = pool[NL - i - 1]  # move non-selected item into vacancy
            else:
                # selected = set()
                selected.clear()
                selected_add = selected.add
                for i in range(flips):
                    j = random.randint(0, NL - 1)
                    while j in selected:
                        j = random.randint(0, NL - 1)
                    selected_add(j)
                    positions[i] = NL_range_tempalte[j]

            for pos in positions:
                wp[pos // l] = wp[pos // l] ^ (1 << (pos % l))

        if flips > 0 or segMut_temp > 1 or _ == 0:
            temp_values = mutational_robustness_nb_smr2_hd(N, wp, viables, l, two_point_distances)
        m_value += 1.0 * temp_values[0]
        avg_hd += 1.0 * temp_values[1]
        nogenotypes += 1.0 * temp_values[2]
        nosegmut += 1.0 * temp_values[3]

    m_value = m_value / measure_generations
    avg_hd = avg_hd / measure_generations
    nogenotypes = nogenotypes/ measure_generations
    nosegmut = nosegmut/ measure_generations


    return m_value, avg_hd, nogenotypes, nosegmut


@nb.njit()
def selrec_mut_cluster_onlyviables(r, initial_generations, measure_generations, N, l, mu, viables, two_point_distances):
    wp = [0] * N
    wp_temp2 = [0] * N
    NL = N * l  # mutation v2 & v3
    #positions = [0]  # mutation v2
    NL_range_tempalte = list(range(NL))  # mutation v3
    selected = {0, 1, 2} #mutation v3

    for _ in range(initial_generations):

        segMut_temp = len(set(wp))
        # if not [wp[0]] * len(wp) == wp:
        if segMut_temp > 1:
            # selection + recombination
            for x3 in range(N):
                wp_temp2[x3] = wp[x3]
            for x3 in range(N):
                if random.random() <= r:
                    randint1 = random.randint(0, N - 1)
                    randint2 = random.randint(0, N - 1)
                    while randint1 == randint2:
                        randint1 = random.randint(0, N - 1)
                        randint2 = random.randint(0, N - 1)
                    a = wp[randint1]
                    b = wp[randint2]
                    c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                    wp_temp2[x3] = c
                else:
                    randint1 = random.randint(0, N - 1)

                    c = wp[randint1]
                    wp_temp2[x3] = c
            for x3 in range(N):
                wp[x3] = wp_temp2[x3]

        # mutation v3
        flips = numpy.random.binomial(NL, mu)
        if flips > 0:
            positions = [0] * flips
            setsize = 21  # size of a small set minus size of an empty list
            if flips > 5:
                setsize += 4 ** math.ceil(math.log(flips * 3) / math.log(4))  # table size for big sets
            if NL <= setsize:
                # An n-length list is smaller than a k-length set
                pool = NL_range_tempalte[:]
                for i in range(flips):  # invariant:  non-selected at [0,n-i)
                    j = random.randint(0, (NL - i) - 1)
                    positions[i] = pool[j]
                    pool[j] = pool[NL - i - 1]  # move non-selected item into vacancy
            else:
                # selected = set()
                selected.clear()
                selected_add = selected.add
                for i in range(flips):
                    j = random.randint(0, NL - 1)
                    while j in selected:
                        j = random.randint(0, NL - 1)
                    selected_add(j)
                    positions[i] = NL_range_tempalte[j]

            for pos in positions:
                wp[pos // l] = wp[pos // l] ^ (1 << (pos % l))

    m_value = 0.0
    avg_hd = 0.0
    nogenotypes = 0.0
    nosegmut = 0.0
    for _ in range(measure_generations):

        segMut_temp = len(set(wp))
        # if not [wp[0]] * len(wp) == wp:
        if segMut_temp > 1:
            # selection + recombination
            for x3 in range(N):
                wp_temp2[x3] = wp[x3]
            for x3 in range(N):
                if random.random() <= r:
                    randint1 = random.randint(0, N - 1)
                    randint2 = random.randint(0, N - 1)
                    while randint1 == randint2:
                        randint1 = random.randint(0, N - 1)
                        randint2 = random.randint(0, N - 1)
                    a = wp[randint1]
                    b = wp[randint2]
                    c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                    wp_temp2[x3] = c
                else:
                    randint1 = random.randint(0, N - 1)

                    c = wp[randint1]
                    wp_temp2[x3] = c
            for x3 in range(N):
                wp[x3] = wp_temp2[x3]

        # mutation v3
        flips = numpy.random.binomial(NL, mu)
        if flips > 0:
            positions = [0] * flips
            setsize = 21  # size of a small set minus size of an empty list
            if flips > 5:
                setsize += 4 ** math.ceil(math.log(flips * 3) / math.log(4))  # table size for big sets
            if NL <= setsize:
                # An n-length list is smaller than a k-length set
                pool = NL_range_tempalte[:]
                for i in range(flips):  # invariant:  non-selected at [0,n-i)
                    j = random.randint(0, (NL - i) - 1)
                    positions[i] = pool[j]
                    pool[j] = pool[NL - i - 1]  # move non-selected item into vacancy
            else:
                # selected = set()
                selected.clear()
                selected_add = selected.add
                for i in range(flips):
                    j = random.randint(0, NL - 1)
                    while j in selected:
                        j = random.randint(0, NL - 1)
                    selected_add(j)
                    positions[i] = NL_range_tempalte[j]

            for pos in positions:
                wp[pos // l] = wp[pos // l] ^ (1 << (pos % l))

        if flips > 0 or segMut_temp > 1 or _ == 0:
            temp_values = mutational_robustness_nb_smr2_hd(N, wp, viables, l, two_point_distances)
        m_value += 1.0 * temp_values[0]
        avg_hd += 1.0 * temp_values[1]
        nogenotypes += 1.0 * temp_values[2]
        nosegmut += 1.0 * temp_values[3]

    m_value = m_value / measure_generations
    avg_hd = avg_hd / measure_generations
    nogenotypes = nogenotypes/ measure_generations
    nosegmut = nosegmut/ measure_generations


    return m_value, avg_hd, nogenotypes, nosegmut


@nb.njit()
def sel_rec_mut_opc_cluster(r, initial_generations, measure_generations, N, l, mu, viables, two_point_distances):
    wp = [0] * N
    wp_temp = [0]
    wp_temp2 = [0] * N
    for _ in range(initial_generations):

        #selection
        del wp_temp[:]
        while len(wp_temp) != N:
            individual = random.randint(0, N - 1)
            if wp[individual] in viables:
                wp_temp.append(wp[individual])
        wp[:] = wp_temp

        #recombination (one-point)
        for x3 in range(N):
            wp_temp2[x3] = wp[x3]
        for x3 in range(N):
            if random.random() <= r:
                randint1 = random.randint(0, N - 1)
                randint2 = random.randint(0, N - 1)
                while randint1 == randint2:
                    randint1 = random.randint(0, N - 1)
                    randint2 = random.randint(0, N - 1)
                a = wp[randint1]
                b = wp[randint2]

                randompos = numpy.random.randint(1, l)
                coinflip = numpy.random.randint(2)
                m = (1 << randompos) - 1
                c = (a if coinflip else b) ^ ((a ^ b) & m)
                """
                randint = random.randint(1, l - 1)
                bitlist_l = [1] * randint + [0] * (l - randint)
                bitlist_r = [0] * randint + [1] * (l - randint)
                l_mask = 0
                r_mask = 0
                for i in range(l):
                    l_mask = (l_mask << 1) | bitlist_l[i]
                    r_mask = (r_mask << 1) | bitlist_r[i]
                if random.random() < 0.5:
                    c = (a & l_mask) | (b & r_mask)
                else:
                    c = (b & l_mask) | (a & r_mask)
                """
                wp_temp2[x3] = c
        for x3 in range(N):
            wp[x3] = wp_temp2[x3]

        #mutation
        for mutation_in_individual in range(N):
            for mutation_at_loci in range(l):
                if random.random() < mu:
                    wp[mutation_in_individual] = wp[mutation_in_individual] ^ (1 << mutation_at_loci)


    m_value = 0.0
    avg_hd = 0.0
    nogenotypes = 0.0
    nosegmut = 0.0
    for _ in range(measure_generations):

        #selection
        del wp_temp[:]
        while len(wp_temp) != N:
            individual = random.randint(0, N - 1)
            if wp[individual] in viables:
                wp_temp.append(wp[individual])
        wp[:] = wp_temp


        #recombination (one-point)
        for x3 in range(N):
            wp_temp2[x3] = wp[x3]
        for x3 in range(N):
            if random.random() <= r:
                randint1 = random.randint(0, N - 1)
                randint2 = random.randint(0, N - 1)
                while randint1 == randint2:
                    randint1 = random.randint(0, N - 1)
                    randint2 = random.randint(0, N - 1)
                a = wp[randint1]
                b = wp[randint2]

                randompos = numpy.random.randint(1, l)
                coinflip = numpy.random.randint(2)
                m = (1 << randompos) - 1
                c = (a if coinflip else b) ^ ((a ^ b) & m)
                """
                randint = random.randint(1, l - 1)
                bitlist_l = [1] * randint + [0] * (l - randint)
                bitlist_r = [0] * randint + [1] * (l - randint)
                l_mask = 0
                r_mask = 0
                for i in range(l):
                    l_mask = (l_mask << 1) | bitlist_l[i]
                    r_mask = (r_mask << 1) | bitlist_r[i]
                if random.random() < 0.5:
                    c = (a & l_mask) | (b & r_mask)
                else:
                    c = (b & l_mask) | (a & r_mask)
                """
                wp_temp2[x3] = c
        for x3 in range(N):
            wp[x3] = wp_temp2[x3]

        #mutation
        for mutation_in_individual in range(N):
            for mutation_at_loci in range(l):
                if random.random() < mu:
                    wp[mutation_in_individual] = wp[mutation_in_individual] ^ (1 << mutation_at_loci)

        temp_values = mutational_robustness_nb_smr2_hd(N, wp, viables, l, two_point_distances)
        m_value += 1.0 * temp_values[0]
        avg_hd += 1.0 * temp_values[1]
        nogenotypes += 1.0 * temp_values[2]
        nosegmut += 1.0 * temp_values[3]

    m_value = m_value / measure_generations
    avg_hd = avg_hd / measure_generations
    nogenotypes = nogenotypes / measure_generations
    nosegmut = nosegmut / measure_generations

    return m_value, avg_hd, nogenotypes, nosegmut



@nb.njit
def explored_genes(N, wp):
    wp_reduced = [0]
    del wp_reduced[:]
    for x3 in range(N):
        if wp[x3] not in wp_reduced:
            wp_reduced.append(wp[x3])
    return wp_reduced


@nb.njit(parallel=True)
def selrec_exploring(rmin, rmax, N, l, mu, datapoints, avg, viables, lethals, log):

    if log is False:
        rlist = numpy.linspace(rmin, rmax, datapoints)
    if log is True:
        rlist_temp = numpy.linspace(numpy.log10(rmin), numpy.log10(rmax), datapoints)
        rlist = numpy.power(10, rlist_temp)


    t_array = numpy.zeros((datapoints, 2))

    for x1 in prange(datapoints):
        t_array[x1, 0] = rlist[x1]
        wp = [0] * N
        wp_temp2 = [0] * N
        wp_reduced = [0] * (2 ** l)
        NL = N * l  # mutation v2 & v3
        # positions = [0]  # mutation v2
        NL_range_tempalte = list(range(NL))  # mutation v3
        selected = {0, 1, 2}  # mutation v3


        for x11 in range(avg):
            viables_copy = viables.copy()
            for x3 in range(N):
                wp[x3] = 0
                wp_temp2[x3] = 0
            for x3 in range(2**l):
                wp_reduced[x3] = 0
            gen_counter = 0
            missing_genotypes = 1
            while missing_genotypes > 0:
                gen_counter += 1
                # selection + recombination
                for x3 in range(N):
                    wp_temp2[x3] = wp[x3]
                for x3 in range(N):
                    if random.random() <= rlist[x1]:
                        randint1 = random.randint(0, N - 1)
                        randint2 = random.randint(0, N - 1)
                        while randint1 == randint2 or wp[randint1] in lethals or wp[randint2] in lethals:
                            randint1 = random.randint(0, N - 1)
                            randint2 = random.randint(0, N - 1)
                        a = wp[randint1]
                        b = wp[randint2]
                        c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                        wp_temp2[x3] = c
                    else:
                        randint1 = random.randint(0, N - 1)
                        while wp[randint1] in lethals:
                            randint1 = random.randint(0, N - 1)
                        c = wp[randint1]
                        wp_temp2[x3] = c
                for x3 in range(N):
                    wp[x3] = wp_temp2[x3]

                """
                # mutation
                for mutation_in_individual in range(N):
                    for mutation_at_loci in range(l):
                        if random.random() < mu:
                            wp[mutation_in_individual] = wp[mutation_in_individual] ^ (1 << mutation_at_loci)
                """
                # mutation v3
                flips = numpy.random.binomial(NL, mu)
                if flips > 0:
                    positions = [0] * flips
                    setsize = 21  # size of a small set minus size of an empty list
                    if flips > 5:
                        setsize += 4 ** math.ceil(math.log(flips * 3) / math.log(4))  # table size for big sets
                    if NL <= setsize:
                        # An n-length list is smaller than a k-length set
                        pool = NL_range_tempalte[:]
                        for i in range(flips):  # invariant:  non-selected at [0,n-i)
                            j = random.randint(0, (NL - i) - 1)
                            positions[i] = pool[j]
                            pool[j] = pool[NL - i - 1]  # move non-selected item into vacancy
                    else:
                        # selected = set()
                        selected.clear()
                        selected_add = selected.add
                        for i in range(flips):
                            j = random.randint(0, NL - 1)
                            while j in selected:
                                j = random.randint(0, NL - 1)
                            selected_add(j)
                            positions[i] = NL_range_tempalte[j]

                    for pos in positions:
                        wp[pos // l] = wp[pos // l] ^ (1 << (pos % l))


                for x3 in range(N):
                    if wp[x3] in viables_copy:
                        viables_copy.remove(wp[x3])
                missing_genotypes = len(viables_copy)
                #print("explored_genotypes", explored_genotypes)
                """
                for x3 in range(N):
                    if wp[x3] not in wp_reduced:
                        wp_reduced.append(wp[x3])
                explored_genotypes = len(wp_reduced)
                """
            #print("r", rlist[x1])
            #print("x1", x1)
            #print("gen_counter", gen_counter)
            #print("\n")
            t_array[x1, 1] += 1.0*gen_counter/avg

    return t_array


@nb.njit(parallel=True)
def selrec_exploring_onlyviables(rmin, rmax, N, l, mu, datapoints, avg, log):

    if log is False:
        rlist = numpy.linspace(rmin, rmax, datapoints)
    if log is True:
        rlist_temp = numpy.linspace(numpy.log10(rmin), numpy.log10(rmax), datapoints)
        rlist = numpy.power(10, rlist_temp)

    t_array = numpy.zeros((datapoints, 2))

    for x1 in range(datapoints):
        t_array[x1, 0] = rlist[x1]
        wp = [0] * N
        wp_temp2 = [0] * N
        wp_reduced = [0] * (2 ** l)
        NL = N * l  # mutation v2 & v3
        NL_range_tempalte = list(range(NL))  # mutation v3
        selected = {0, 1, 2}  # mutation v3

        for x11 in range(avg):
            for x3 in range(N):
                wp[x3] = 0
                wp_temp2[x3] = 0
            for x3 in range(2**l):
                wp_reduced[x3] = 0
            gen_counter = 0
            explored_genotypes = 0
            while explored_genotypes < 2**l:
                gen_counter += 1
                # selection + recombination
                for x3 in range(N):
                    wp_temp2[x3] = wp[x3]
                for x3 in range(N):
                    if random.random() <= rlist[x1]:
                        randint1 = random.randint(0, N - 1)
                        randint2 = random.randint(0, N - 1)
                        while randint1 == randint2:
                            randint1 = random.randint(0, N - 1)
                            randint2 = random.randint(0, N - 1)
                        a = wp[randint1]
                        b = wp[randint2]
                        c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                        wp_temp2[x3] = c
                    else:
                        randint1 = random.randint(0, N - 1)
                        c = wp[randint1]
                        wp_temp2[x3] = c
                for x3 in range(N):
                    wp[x3] = wp_temp2[x3]

                """
                # mutation
                for mutation_in_individual in range(N):
                    for mutation_at_loci in range(l):
                        if random.random() < mu:
                            wp[mutation_in_individual] = wp[mutation_in_individual] ^ (1 << mutation_at_loci)
                """
                # mutation v3
                flips = numpy.random.binomial(NL, mu)
                if flips > 0:
                    positions = [0] * flips
                    setsize = 21  # size of a small set minus size of an empty list
                    if flips > 5:
                        setsize += 4 ** math.ceil(math.log(flips * 3) / math.log(4))  # table size for big sets
                    if NL <= setsize:
                        # An n-length list is smaller than a k-length set
                        pool = NL_range_tempalte[:]
                        for i in range(flips):  # invariant:  non-selected at [0,n-i)
                            j = random.randint(0, (NL - i) - 1)
                            positions[i] = pool[j]
                            pool[j] = pool[NL - i - 1]  # move non-selected item into vacancy
                    else:
                        # selected = set()
                        selected.clear()
                        selected_add = selected.add
                        for i in range(flips):
                            j = random.randint(0, NL - 1)
                            while j in selected:
                                j = random.randint(0, NL - 1)
                            selected_add(j)
                            positions[i] = NL_range_tempalte[j]

                    for pos in positions:
                        wp[pos // l] = wp[pos // l] ^ (1 << (pos % l))


                for x3 in range(N):
                    wp_reduced[wp[x3]] = 1
                explored_genotypes = 0
                for x3 in range(2**l):
                    explored_genotypes += wp_reduced[x3]

                """
                for x3 in range(N):
                    if wp[x3] not in wp_reduced:
                        wp_reduced.append(wp[x3])
                explored_genotypes = len(wp_reduced)
                """
            #print("r", rlist[x1])
            #print("x1", x1)
            #print("gen_counter", gen_counter)
            #print("\n")
            t_array[x1, 1] += 1.0*gen_counter/avg

    return t_array


@nb.njit(parallel=True)
def selrec_exploring_antipodal(rmin, rmax, N, l, mu, datapoints, avg, log):

    if log is False:
        rlist = numpy.linspace(rmin, rmax, datapoints)
    if log is True:
        rlist_temp = numpy.linspace(numpy.log10(rmin), numpy.log10(rmax), datapoints)
        rlist = numpy.power(10, rlist_temp)


    t_array = numpy.zeros((datapoints, 2))

    for x1 in prange(datapoints):
        t_array[x1, 0] = rlist[x1]
        wp = [0] * N
        wp_temp2 = [0] * N
        NL = N * l  # mutation v2 & v3
        # positions = [0]  # mutation v2
        NL_range_tempalte = list(range(NL))  # mutation v3
        selected = {0, 1, 2}  # mutation v3

        for x11 in range(avg):
            gen_counter = 0
            antipodal_found = False
            for x3 in range(N):
                wp[x3] = 0
                wp_temp2[x3] = 0
            while antipodal_found == False:
                gen_counter += 1
                # selection + recombination
                for x3 in range(N):
                    wp_temp2[x3] = wp[x3]
                for x3 in range(N):
                    if random.random() <= rlist[x1]:
                        randint1 = random.randint(0, N - 1)
                        randint2 = random.randint(0, N - 1)
                        while randint1 == randint2:
                            randint1 = random.randint(0, N - 1)
                            randint2 = random.randint(0, N - 1)
                        a = wp[randint1]
                        b = wp[randint2]
                        c = a ^ ((a ^ b) & random.randint(0, (1 << l) - 1))
                        wp_temp2[x3] = c
                    else:
                        randint1 = random.randint(0, N - 1)
                        c = wp[randint1]
                        wp_temp2[x3] = c
                for x3 in range(N):
                    wp[x3] = wp_temp2[x3]

                """
                # mutation
                for mutation_in_individual in range(N):
                    for mutation_at_loci in range(l):
                        if random.random() < mu:
                            wp[mutation_in_individual] = wp[mutation_in_individual] ^ (1 << mutation_at_loci)
                """
                # mutation v3
                flips = numpy.random.binomial(NL, mu)
                if flips > 0:
                    positions = [0] * flips
                    setsize = 21  # size of a small set minus size of an empty list
                    if flips > 5:
                        setsize += 4 ** math.ceil(math.log(flips * 3) / math.log(4))  # table size for big sets
                    if NL <= setsize:
                        # An n-length list is smaller than a k-length set
                        pool = NL_range_tempalte[:]
                        for i in range(flips):  # invariant:  non-selected at [0,n-i)
                            j = random.randint(0, (NL - i) - 1)
                            positions[i] = pool[j]
                            pool[j] = pool[NL - i - 1]  # move non-selected item into vacancy
                    else:
                        # selected = set()
                        selected.clear()
                        selected_add = selected.add
                        for i in range(flips):
                            j = random.randint(0, NL - 1)
                            while j in selected:
                                j = random.randint(0, NL - 1)
                            selected_add(j)
                            positions[i] = NL_range_tempalte[j]

                    for pos in positions:
                        wp[pos // l] = wp[pos // l] ^ (1 << (pos % l))


                for x3 in range(N):
                    if wp[x3] == 2**l-1:
                        antipodal_found = True


            #print("r", rlist[x1])
            #print("x1", x1)
            #print("gen_counter", gen_counter)
            #print("\n")
            t_array[x1, 1] += 1.0*gen_counter/avg
    return t_array
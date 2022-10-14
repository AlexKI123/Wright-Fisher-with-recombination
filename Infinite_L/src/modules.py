import functools
import operator
import random
from collections import Counter
from scipy import stats
import scipy.special
import numpy as np


def functools_reduce_iconcat(a):
    return functools.reduce(operator.iconcat, a, [])


def selection(wp , N):
    wp_temp = random.choices(wp, k=N)
    wp = [x.copy() for x in wp_temp]
    return wp


def selection_vl(wp , N, lethals):
    wp_temp = [x.copy() for x in wp]
    for x1 in range(N):
        randint1 = random.randint(0, N - 1)
        while tuple(sorted(wp[randint1])) in lethals:
            randint1 = random.randint(0, N - 1)
        c = wp[randint1]
        wp_temp[x1] = c.copy()
    wp = [x.copy() for x in wp_temp]
    return wp


def mutation(wp, N, U, mutation_counter):
    for individual in range(N):
        if random.random() < U:
            wp[individual].add(mutation_counter + 1)
            mutation_counter = mutation_counter + 1
    return wp, mutation_counter


def mutation_vl_poisson(wp, N, U, p, mutation_counter, viables, lethals):
    for individual in range(N):
        mutations_U = np.random.poisson(U)
        if mutations_U > 0:
            for i in range(mutations_U):
                mutation_counter = mutation_counter + 1
                wp[individual].add(mutation_counter)
                if p > random.random():
                    viables.add(tuple(sorted(wp[individual])))
                else:
                   lethals.add(tuple(sorted(wp[individual])))
    return wp, mutation_counter, viables, lethals


def recombination(wp, N, r):
    wp_temp = [x.copy() for x in wp]
    for x1 in range(N):
        if random.random() <= r:
            randint1 = random.randint(0, N - 1)
            randint2 = random.randint(0, N - 1)
            while randint1 == randint2:
                randint1 = random.randint(0, N - 1)
                randint2 = random.randint(0, N - 1)
            a = wp[randint1]
            # a = wp[x1]
            b = wp[randint2]
            c = a & b
            for x in a ^ b:
                if random.random() < 0.5:
                    c.add(x)
            wp_temp[x1] = c.copy()
    wp = [x.copy() for x in wp_temp]
    return wp


def recombination_vl(wp, N, r, p, viables, lethals):
    viable_counter = 0
    lethal_counter = 0
    novel_counter = 0
    recomb_counter = 0
    wp_temp = [x.copy() for x in wp]
    for x1 in range(N):
        if random.random() <= r:
            recomb_counter += 1
            randint1 = random.randint(0, N - 1)
            randint2 = random.randint(0, N - 1)
            while randint1 == randint2:
                randint1 = random.randint(0, N - 1)
                randint2 = random.randint(0, N - 1)
            a = wp[randint1]
            b = wp[randint2]
            c = a & b
            for x in a ^ b:
                if random.random() < 0.5:
                    c.add(x)
            wp_temp[x1] = c.copy()
            if tuple(sorted(wp_temp[x1])) in viables:
                viable_counter += 1
            elif tuple(sorted(wp_temp[x1])) in lethals:
                lethal_counter += 1
            else:
                novel_counter += 1
                if p > random.random():
                    viable_counter += 1
                    viables.add(tuple(sorted(wp_temp[x1])))
                else:
                    lethal_counter += 1
                    lethals.add(tuple(sorted(wp_temp[x1])))
    wp = [x.copy() for x in wp_temp]
    return wp, viables, lethals, viable_counter, lethal_counter, novel_counter, recomb_counter


# one generation with concurrent recombination -- only viables
def selrec(wp, N, r):
    wp_temp = [x.copy() for x in wp]
    for x1 in range(N):
        if random.random() <= r:
            randint1 = random.randint(0, N - 1)
            randint2 = random.randint(0, N - 1)
            while randint1 == randint2:
                randint1 = random.randint(0, N - 1)
                randint2 = random.randint(0, N - 1)
            a = wp[randint1]
            b = wp[randint2]
            c = a & b
            for x in a ^ b:
                if random.random() < 0.5:
                    c.add(x)
            wp_temp[x1] = c.copy()
        else:
            randint1 = random.randint(0, N - 1)
            c = wp[randint1]
            wp_temp[x1] = c.copy()
    wp = [x.copy() for x in wp_temp]
    return wp


# one generation with concurrent recombination and lethal genotypes
def selrec_vl(wp, N, r, p, viables, lethals):
    wp_temp = [x.copy() for x in wp]
    for x1 in range(N):
        if random.random() <= r:
            randint1 = random.randint(0, N - 1)
            randint2 = random.randint(0, N - 1)
            while randint1 == randint2 or tuple(sorted(wp[randint1])) in lethals or tuple(sorted(wp[randint2])) in lethals:
                randint1 = random.randint(0, N - 1)
                randint2 = random.randint(0, N - 1)
            a = wp[randint1]
            b = wp[randint2]
            c = a & b
            for x in a ^ b:
                if random.random() < 0.5:
                    c.add(x)
            wp_temp[x1] = c.copy()
            if tuple(sorted(wp_temp[x1])) not in lethals and tuple(sorted(wp_temp[x1])) not in viables:
                if p > random.random():
                    viables.add(tuple(sorted(wp_temp[x1])))
                else:
                    lethals.add(tuple(sorted(wp_temp[x1])))
        else:
            randint1 = random.randint(0, N - 1)
            while tuple(sorted(wp[randint1])) in lethals:
                randint1 = random.randint(0, N - 1)
            c = wp[randint1]
            wp_temp[x1] = c.copy()
    wp = [x.copy() for x in wp_temp]
    return wp, viables, lethals


# one generation with concurrent recombination and lethal genotypes + counting viable recombination events
def selrec_vl_v2(wp, N, r, p, viables, lethals):
    viable_counter = 0
    lethal_counter = 0
    novel_counter = 0
    recomb_counter = 0
    wp_temp = [x.copy() for x in wp]
    for x1 in range(N):
        if random.random() <= r:
            recomb_counter += 1
            randint1 = random.randint(0, N - 1)
            randint2 = random.randint(0, N - 1)
            while randint1 == randint2 or tuple(sorted(wp[randint1])) in lethals or tuple(sorted(wp[randint2])) in lethals:
                randint1 = random.randint(0, N - 1)
                randint2 = random.randint(0, N - 1)
            a = wp[randint1]
            b = wp[randint2]
            c = a & b
            for x in a ^ b:
                if random.random() < 0.5:
                    c.add(x)
            wp_temp[x1] = c.copy()
            if tuple(sorted(wp_temp[x1])) in viables:
                viable_counter += 1
            elif tuple(sorted(wp_temp[x1])) in lethals:
                lethal_counter += 1
            else:
                novel_counter += 1
                if p > random.random():
                    viable_counter += 1
                    viables.add(tuple(sorted(wp_temp[x1])))
                else:
                    lethal_counter += 1
                    lethals.add(tuple(sorted(wp_temp[x1])))
        else:
            randint1 = random.randint(0, N - 1)
            while tuple(sorted(wp[randint1])) in lethals:
                randint1 = random.randint(0, N - 1)
            c = wp[randint1]
            wp_temp[x1] = c.copy()
    wp = [x.copy() for x in wp_temp]

    return wp, viables, lethals, viable_counter, lethal_counter, novel_counter, recomb_counter


def clean_fixed_mutations_m7(wp, N, fixed_mutations, viables, lethals, explored_genotypes, explored_viable_genotypes):
    new_fixed_mutations = set.intersection(*wp)
    if len(new_fixed_mutations) > 0:
        fixed_mutations = fixed_mutations.union(new_fixed_mutations)
        for x in range(N):
            wp[x] = wp[x] - fixed_mutations

        #unique_mutations = set(functools_reduce_iconcat([list(item) for item in wp]))

        new_viables = set()
        new_lethals = set()
        for val in viables:
            valnew = tuple(sorted([x for x in val if x not in new_fixed_mutations]))
            if len(valnew) == len(val)-len(new_fixed_mutations):
                new_viables.add(valnew)

        for val in lethals:
            valnew = tuple(sorted([x for x in val if x not in new_fixed_mutations]))
            if len(valnew) == len(val)-len(new_fixed_mutations):
                new_lethals.add(valnew)

        new_explored = set()
        counter = 0
        for val in explored_genotypes:
            valnew = tuple(sorted([x for x in val if x not in new_fixed_mutations]))
            if len(valnew) == len(val)-len(new_fixed_mutations):
                new_explored.add(valnew)
            else:
                counter += 1

        new_viable_explored = set()
        viable_counter = 0
        for val in explored_viable_genotypes:
            valnew = tuple(sorted([x for x in val if x not in new_fixed_mutations]))
            if len(valnew) == len(val)-len(new_fixed_mutations):
                new_viable_explored.add(valnew)
            else:
                viable_counter += 1

    else:
        counter = 0
        viable_counter = 0
        new_viables = viables
        new_lethals = lethals
        new_explored = explored_genotypes
        new_viable_explored = explored_viable_genotypes

    return wp, fixed_mutations, new_viables, new_lethals, new_explored, new_viable_explored, counter, viable_counter


# concurrent recombination for #generations
def selrec_r_U_cluster(r, U, N, generations, discarded_elements, p):

    avg_hd_divider = scipy.special.binom(N, 2)

    mutation_counter = 0

    explored_genotypes_list = [0] * generations
    explored_viable_genotypes_list = [0] * generations

    number_distinct_genotypes = [0] * generations
    number_viable_distinct_genotypes = [0] * generations

    number_segregating_mutations = [0] * generations
    avg_distance = [0] * generations
    max_distance = [0] * generations
    mean_Fitness = [0] * generations

    number_fixed_mutations = [0] * generations

    avg_dist_fixed = [0] * generations


    viable_counter = 0
    lethal_counter = 0
    novel_counter = 0
    recomb_counter = 0

    wp = [set() for _ in range(N)]
    fixed_mutations = set()

    explored_genotypes = {()}
    explored_viable_genotypes = {()}

    explored_genotypes_counter = 0
    explored_viable_genotypes_counter = 0


    viables = {()}
    lethals = set()


    for gen in range(generations):
        #print("gen", gen)
        # Evolution
        #wp, viables, lethals = selrec_vl(wp, N, r, p, viables, lethals)
        wp, viables, lethals, viable_counter_temp, lethal_counter_temp, novel_counter_temp, recomb_counter_temp = selrec_vl_v2(wp, N, r, p, viables, lethals)
        wp, mutation_counter, viables, lethals = mutation_vl_poisson(wp, N, U, p, mutation_counter, viables, lethals)


        # clean fixed mutations
        wp, fixed_mutations, viables, lethals, explored_genotypes, explored_viable_genotypes, explored_genotypes_counter_temp, explored_viable_genotypes_counter_temp = clean_fixed_mutations_m7(wp, N, fixed_mutations, viables, lethals, explored_genotypes, explored_viable_genotypes)
        explored_genotypes_counter += explored_genotypes_counter_temp
        explored_viable_genotypes_counter += explored_viable_genotypes_counter_temp


        # Analysis
        viable_counter += viable_counter_temp
        lethal_counter += lethal_counter_temp
        novel_counter += novel_counter_temp
        recomb_counter += recomb_counter_temp


        uniques = Counter(tuple(sorted(item)) for item in wp)
        distinct_genotypes = uniques.keys()
        distinct_and_viable_genotypes = {x for x in distinct_genotypes if x in viables}
        explored_genotypes.update(distinct_genotypes)
        explored_viable_genotypes.update(distinct_and_viable_genotypes)
        number_distinct_genotypes[gen] = len(distinct_genotypes)
        number_viable_distinct_genotypes[gen] = len(distinct_and_viable_genotypes)
        explored_genotypes_list[gen] = len(explored_genotypes) + explored_genotypes_counter
        explored_viable_genotypes_list[gen] = len(explored_viable_genotypes) + explored_viable_genotypes_counter
        number_segregating_mutations[gen] = len(set.union(*wp))
        number_fixed_mutations[gen] = len(fixed_mutations)
        ###mean_Fitness[gen] = 1.0*sum([True for x in wp if tuple(sorted(x)) in viables])/N
        mean_Fitness[gen] = 1.0*sum([uniques[x] for x in distinct_genotypes if x in viables]) / N
        avg_dist_fixed[gen] = 1.0*sum([len(wp[x]) for x in range(N)]) / N


        avg_population_hamdist = 0
        distinct_genotypes_list = list(distinct_genotypes)
        frequency_list = list(uniques.values())
        for x1 in range(len(uniques.keys())-1):
            for x2 in range(x1 + 1, len(uniques.keys())):
                h_d_temp = len(set(distinct_genotypes_list[x1]) ^ set(distinct_genotypes_list[x2]))
                avg_population_hamdist += 1.0 * h_d_temp * frequency_list[x1] * frequency_list[x2]
                if h_d_temp > max_distance[gen]:
                    max_distance[gen] = h_d_temp
        avg_population_hamdist = avg_population_hamdist / avg_hd_divider
        avg_distance[gen] = avg_population_hamdist


        ###distance_temp = list(len(x ^ y) for x, y in comb(wp, 2))
        ###avg_distance[gen] = sum(distance_temp) * 2 / (N * (N - 1))
        ###max_distance[gen] = max(distance_temp)

    if recomb_counter > 0:
        viable_fraction = viable_counter/recomb_counter
        lethal_fraction = lethal_counter/recomb_counter
        novel_fraction = novel_counter/recomb_counter
    else:
        viable_fraction = 1
        lethal_fraction = 0
        novel_fraction = 0


    y_values = explored_genotypes_list[discarded_elements:]
    x_values = list(range(len(y_values)))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
    rate = slope

    y_values_viable = explored_viable_genotypes_list[discarded_elements:]
    x_values_viable = list(range(len(y_values_viable)))
    slope_viable, intercept, r_value, p_value, std_err = stats.linregress(x_values_viable, y_values_viable)
    rate_viable = slope_viable

    y_fixed = number_fixed_mutations[discarded_elements:]
    x_fixed = list(range(len(y_fixed)))
    slope_fixed, intercept, r_value, p_value, std_err = stats.linregress(x_fixed, y_fixed)
    rate_fixed = slope_fixed

    dist_ge = sum(number_distinct_genotypes[discarded_elements:]) / len(number_distinct_genotypes[discarded_elements:])
    dist_viable_ge = sum(number_viable_distinct_genotypes[discarded_elements:]) / len(number_viable_distinct_genotypes[discarded_elements:])

    seg_mut = sum(number_segregating_mutations[discarded_elements:]) / len(number_segregating_mutations[discarded_elements:])
    mean_F = sum(mean_Fitness[discarded_elements:]) / len(mean_Fitness[discarded_elements:])

    rate2 = 1.0 * (explored_genotypes_list[-1] - explored_genotypes_list[discarded_elements]) / (generations - discarded_elements)
    rate2_viable = 1.0 * (explored_viable_genotypes_list[-1] - explored_viable_genotypes_list[discarded_elements]) / (generations - discarded_elements)

    avg_dist = sum(avg_distance[discarded_elements:]) / len(avg_distance[discarded_elements:])
    max_dist = sum(max_distance[discarded_elements:]) / len(max_distance[discarded_elements:])


    avg_dist_fi = sum(avg_dist_fixed[discarded_elements:]) / len(avg_dist_fixed[discarded_elements:])

    return rate, rate2, rate_viable, rate2_viable, rate_fixed, avg_dist, max_dist, dist_ge, dist_viable_ge, seg_mut, mean_F, viable_fraction, lethal_fraction, novel_fraction, avg_dist_fi


# simple successive recombination for #generations
def sel_rec_r_U_cluster(r, U, N, generations, discarded_elements, p):

    avg_hd_divider = scipy.special.binom(N, 2)

    mutation_counter = 0

    explored_genotypes_list = [0] * generations
    explored_viable_genotypes_list = [0] * generations

    number_distinct_genotypes = [0] * generations
    number_viable_distinct_genotypes = [0] * generations

    number_segregating_mutations = [0] * generations
    avg_distance = [0] * generations
    max_distance = [0] * generations
    mean_Fitness = [0] * generations

    number_fixed_mutations = [0] * generations

    avg_dist_fixed = [0] * generations


    viable_counter = 0
    lethal_counter = 0
    novel_counter = 0
    recomb_counter = 0

    wp = [set() for _ in range(N)]
    fixed_mutations = set()

    explored_genotypes = {()}
    explored_viable_genotypes = {()}

    explored_genotypes_counter = 0
    explored_viable_genotypes_counter = 0


    viables = {()}
    lethals = set()


    for gen in range(generations):
        # Evolution
        #wp, viables, lethals = selrec_vl(wp, N, r, p, viables, lethals)
        wp = selection_vl(wp, N, lethals)
        wp, viables, lethals, viable_counter_temp, lethal_counter_temp, novel_counter_temp, recomb_counter_temp = recombination_vl(wp, N, r, p, viables, lethals)
        #wp, viables, lethals, viable_counter_temp, lethal_counter_temp, novel_counter_temp, recomb_counter_temp = selrec_vl_v2(wp, N, r, p, viables, lethals)
        wp, mutation_counter, viables, lethals = mutation_vl_poisson(wp, N, U, p, mutation_counter, viables, lethals)

        # clean fixed mutations
        wp, fixed_mutations, viables, lethals, explored_genotypes, explored_viable_genotypes, explored_genotypes_counter_temp, explored_viable_genotypes_counter_temp = clean_fixed_mutations_m7(wp, N, fixed_mutations, viables, lethals, explored_genotypes, explored_viable_genotypes)
        explored_genotypes_counter += explored_genotypes_counter_temp
        explored_viable_genotypes_counter += explored_viable_genotypes_counter_temp


        # Analysis
        viable_counter += viable_counter_temp
        lethal_counter += lethal_counter_temp
        novel_counter += novel_counter_temp
        recomb_counter += recomb_counter_temp


        uniques = Counter(tuple(sorted(item)) for item in wp)
        distinct_genotypes = uniques.keys()
        distinct_and_viable_genotypes = {x for x in distinct_genotypes if x in viables}
        explored_genotypes.update(distinct_genotypes)
        explored_viable_genotypes.update(distinct_and_viable_genotypes)
        number_distinct_genotypes[gen] = len(distinct_genotypes)
        number_viable_distinct_genotypes[gen] = len(distinct_and_viable_genotypes)
        explored_genotypes_list[gen] = len(explored_genotypes) + explored_genotypes_counter
        explored_viable_genotypes_list[gen] = len(explored_viable_genotypes) + explored_viable_genotypes_counter
        number_segregating_mutations[gen] = len(set.union(*wp))
        number_fixed_mutations[gen] = len(fixed_mutations)
        ###mean_Fitness[gen] = 1.0*sum([True for x in wp if tuple(sorted(x)) in viables])/N
        mean_Fitness[gen] = 1.0*sum([uniques[x] for x in distinct_genotypes if x in viables]) / N

        avg_dist_fixed[gen] = 1.0*sum([len(wp[x]) for x in range(N)]) / N



        avg_population_hamdist = 0
        distinct_genotypes_list = list(distinct_genotypes)
        frequency_list = list(uniques.values())
        for x1 in range(len(uniques.keys())-1):
            for x2 in range(x1 + 1, len(uniques.keys())):
                h_d_temp = len(set(distinct_genotypes_list[x1]) ^ set(distinct_genotypes_list[x2]))
                avg_population_hamdist += 1.0 * h_d_temp * frequency_list[x1] * frequency_list[x2]
                if h_d_temp > max_distance[gen]:
                    max_distance[gen] = h_d_temp
        avg_population_hamdist = avg_population_hamdist / avg_hd_divider
        avg_distance[gen] = avg_population_hamdist


        ###distance_temp = list(len(x ^ y) for x, y in comb(wp, 2))
        ###avg_distance[gen] = sum(distance_temp) * 2 / (N * (N - 1))
        ###max_distance[gen] = max(distance_temp)

    if recomb_counter > 0:
        viable_fraction = viable_counter/recomb_counter
        lethal_fraction = lethal_counter/recomb_counter
        novel_fraction = novel_counter/recomb_counter
    else:
        viable_fraction = 1
        lethal_fraction = 0
        novel_fraction = 0


    y_values = explored_genotypes_list[discarded_elements:]
    x_values = list(range(len(y_values)))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
    rate = slope

    y_values_viable = explored_viable_genotypes_list[discarded_elements:]
    x_values_viable = list(range(len(y_values_viable)))
    slope_viable, intercept, r_value, p_value, std_err = stats.linregress(x_values_viable, y_values_viable)
    rate_viable = slope_viable

    y_fixed = number_fixed_mutations[discarded_elements:]
    x_fixed = list(range(len(y_fixed)))
    slope_fixed, intercept, r_value, p_value, std_err = stats.linregress(x_fixed, y_fixed)
    rate_fixed = slope_fixed

    dist_ge = sum(number_distinct_genotypes[discarded_elements:]) / len(number_distinct_genotypes[discarded_elements:])
    dist_viable_ge = sum(number_viable_distinct_genotypes[discarded_elements:]) / len(number_viable_distinct_genotypes[discarded_elements:])

    seg_mut = sum(number_segregating_mutations[discarded_elements:]) / len(number_segregating_mutations[discarded_elements:])
    mean_F = sum(mean_Fitness[discarded_elements:]) / len(mean_Fitness[discarded_elements:])

    rate2 = 1.0 * (explored_genotypes_list[-1] - explored_genotypes_list[discarded_elements]) / (generations - discarded_elements)
    rate2_viable = 1.0 * (explored_viable_genotypes_list[-1] - explored_viable_genotypes_list[discarded_elements]) / (generations - discarded_elements)

    avg_dist = sum(avg_distance[discarded_elements:]) / len(avg_distance[discarded_elements:])
    max_dist = sum(max_distance[discarded_elements:]) / len(max_distance[discarded_elements:])
    avg_dist_fi = sum(avg_dist_fixed[discarded_elements:]) / len(avg_dist_fixed[discarded_elements:])

    return rate, rate2, rate_viable, rate2_viable, rate_fixed, avg_dist, max_dist, dist_ge, dist_viable_ge, seg_mut, mean_F, viable_fraction, lethal_fraction, novel_fraction, avg_dist_fi







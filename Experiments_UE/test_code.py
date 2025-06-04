import data_read_perturb as drp

def error_cal(true_list,estimate_list):
    return sum([(true_list[i] - estimate_list[i]) ** 2 for i in range(len(true_list))]) / len(true_list)

def UE_test(true_list,num1,epsilon_list,test_time):

    err_naive = 0
    err_baseline = 0
    err_best = 0

    for i in range(test_time):
        result1 = drp.UE_reuse(true_list, num1, epsilon_list)

        err_naive += result1[0]
        err_baseline += result1[1]
        err_best += drp.UE(true_list, num1, sum(epsilon_list))[0]


    print(err_naive/test_time,"\t",err_baseline/test_time,"\t",err_best/test_time)
    return 0


def UE_partition_test(true_list,epsilon,n,test_time):

    err_sum_list = [0 for i in range(n)]

    for t in range(test_time):

        for i in range(1, n + 1):
            shuffled_list=[int(value*i/n) for value in true_list]

            err_temp = error_cal(true_list,[(int(value_2*n/i)) for value_2 in drp.UE_partition(shuffled_list, epsilon)])

            err_sum_list[i - 1] += err_temp

    return [q / test_time for q in err_sum_list]


def k_test(true_list,epsilon,n,test_time):

    err_sum_list = [0 for i in range(n)]

    for t in range(test_time):

        for i in range(1, n + 1):
            estimated_temp = [0 for i in range(len(true_list))]

            epsilon_temp = epsilon / i
            for j in range(1, i + 1):

                new_es=drp.UE(true_list,sum(true_list),epsilon_temp)[1]
                estimated_temp = [estimated_temp[t]+new_es[t] for t in range(len(true_list))]

            estimated_temp=[q/i for q in estimated_temp]
            err_sum_list[i - 1] += error_cal(true_list,estimated_temp)

    return [q / test_time for q in err_sum_list]
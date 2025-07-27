import data_perturb as drp
import random

import pandas as pd



def SR_test(true_list,epsilon_list,test_time):

    err_baseline = 0
    err_best = 0

    for i in range(test_time):
        result1 = drp.SR_reuse(true_list, epsilon_list)

        err_baseline += result1[1]
        err_best += drp.SR(true_list, sum(epsilon_list))[0]

    print(err_baseline/test_time,"\t",err_best/test_time)
    return 0



def k_test(data_list,epsilon,n,test_time):
    result_true = sum(data_list)/len(data_list)

    err_sum_list = [0 for i in range(n)]

    for t in range(test_time):

        for i in range(1,n+1):
            estimated_temp = 0

            epsilon_temp = epsilon / i
            for j in range(1,i+1):

                estimated_temp+=drp.SR(data_list, epsilon_temp)[1]

            err_sum_list[i-1]+=abs(estimated_temp/i-result_true)/result_true

    return [q/test_time for q in err_sum_list]
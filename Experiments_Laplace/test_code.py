import numpy as np
import polynomial as pol
import random
def x_2_star_test(data_list,epsilon_list,test_time):
    sum_epsilon = sum(epsilon_list)

    result_true = sum([(i ** 2 - i) / 2 for i in data_list])

    err_naive = 0
    err_base = 0
    err_best = 0

    for i in range(test_time):
        noisy_mar = [[i + np.random.laplace(0, 2 / epsilon) for i in data_list] for epsilon in epsilon_list]
        noisy_best = [i + np.random.laplace(0, 2 / sum_epsilon) for i in data_list]

        result_naive = (pol.x2_cal(noisy_mar[0], 2 / max(epsilon_list)))

        result_base = (pol.x2_cal_baseline(noisy_mar, epsilon_list))

        result_best = (pol.x2_cal(noisy_best, 2 / sum_epsilon))

        err_naive += abs(result_naive - result_true) / result_true
        err_base += abs(result_base - result_true) / result_true
        err_best += abs(result_best - result_true) / result_true

    return err_naive/test_time,err_base/test_time,err_best/test_time



def partition_test(data_list,epsilon,n,test_time):
    result_true=sum([(i ** 2 - i) / 2 for i in data_list])

    err_sum_list=[0 for i in range(n)]

    for t in range(test_time):

        shuffled_list = data_list[:]
        random.shuffle(shuffled_list)

        shuffled_list = [i + np.random.laplace(0, 2 / epsilon) for i in shuffled_list]

        for i in range(1,n+1):

            num=int(i/n*len(shuffled_list))

            err_temp=abs(pol.x2_cal(shuffled_list[:num],2/epsilon)*n/i-result_true)/result_true

            #print(abs(pol.x2_cal(shuffled_list,2/epsilon)-result_true)/result_true)

            err_sum_list[i-1]+=err_temp

    return [q/test_time for q in err_sum_list]




def k_test(data_list,epsilon,n,test_time):
    result_true = sum([(i ** 2 - i) / 2 for i in data_list])

    err_sum_list = [0 for i in range(n)]

    for t in range(test_time):

        for i in range(1,n+1):
            estimated_temp = 0

            epsilon_temp = epsilon / i
            for j in range(1,i+1):

                noisy_list=[i+np.random.laplace(0, 2 / epsilon_temp) for i in data_list]

                estimated_temp+=pol.x2_cal(noisy_list,2/epsilon_temp)

            err_sum_list[i-1]+=abs(estimated_temp/i-result_true)/result_true

    return [q/test_time for q in err_sum_list]

import numpy as np
import polynomial as pol
import random
def sigma_cal(epsilon_list,deleta_list):
    sigma_list=[]
    for i in range(len(epsilon_list)):
        sigma_list.append((2*np.log(1.25/deleta_list[i]))**0.5*2/epsilon_list[i])
    return sigma_list

def x_2star_test(data_list,epsilon_list,delta_list,test_time):
    sigma_list = sigma_cal(epsilon_list, delta_list)
    sig_best = (2 * np.log(1.25 / sum(delta_list))) ** 0.5 * 2 / sum(epsilon_list)

    result_true = sum([(i ** 2-i)/2 for i in data_list])


    #print(result_true)

    err_naive = 0
    err_base = 0
    err_best = 0

    for i in range(test_time):
        noisy_mar = [[i + np.random.normal(0, sig) for i in data_list] for sig in sigma_list]
        noisy_best = [i + np.random.normal(0, sig_best) for i in data_list]

        result_naive = (pol.x2_cal(noisy_mar[0], sigma_list[0]))

        result_base = (pol.x2_cal_baseline(noisy_mar, sigma_list))

        result_best = (pol.x2_cal(noisy_best, sig_best))

        err_naive += abs(result_naive - result_true) / result_true
        err_base += abs(result_base - result_true) / result_true
        err_best += abs(result_best - result_true) / result_true

    return (err_naive/test_time,err_base/test_time,err_best/test_time)


def partition_test(data_list,epsilon,delta,n,test_time):
    result_true=sum([(i ** 2 - i) / 2 for i in data_list])

    sig=(2 * np.log(1.25 / delta)) ** 0.5 * 2 / epsilon

    err_sum_list=[0 for i in range(n)]

    for t in range(test_time):

        shuffled_list = data_list[:]
        random.shuffle(shuffled_list)

        shuffled_list = [i + np.random.normal(0, sig) for i in shuffled_list]

        for i in range(1,n+1):

            num=int(i/n*len(shuffled_list))

            err_temp=abs(pol.x2_cal(shuffled_list[:num],sig)*n/i-result_true)/result_true

            #print(abs(pol.x2_cal(shuffled_list,2/epsilon)-result_true)/result_true)

            err_sum_list[i-1]+=err_temp

    return [q/test_time for q in err_sum_list]


def k_test(data_list,epsilon,delta,n,test_time):
    result_true = sum([(i ** 2 - i) / 2 for i in data_list])

    err_sum_list = [0 for i in range(n)]

    for t in range(test_time):

        for i in range(1,n+1):
            estimated_temp = 0

            epsilon_temp = epsilon / i
            for j in range(1,i+1):

                sig = (2 * np.log(1.25 / delta)) ** 0.5 * 2 / epsilon_temp

                noisy_list=[i+np.random.normal(0,sig) for i in data_list]

                estimated_temp+=pol.x2_cal(noisy_list,sig)

            err_sum_list[i-1]+=abs(estimated_temp/i-result_true)/result_true

    return [q/test_time for q in err_sum_list]


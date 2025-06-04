import data_perturb as drp
import random


def SR_test(true_list,epsilon_list,test_time):

    err_naive = 0
    err_baseline = 0
    err_best = 0

    for i in range(test_time):
        result1 = drp.SR_reuse(true_list, epsilon_list)

        err_naive += result1[0]
        err_baseline += result1[1]
        err_best += drp.SR(true_list, sum(epsilon_list))[0]

    print(err_naive/test_time,"\t",err_baseline/test_time,"\t",err_best/test_time)
    return 0


def partition_test(data_list,epsilon,n,test_time):
    mean_true=sum(data_list)/len(data_list)
    err_sum_list=[0 for i in range(n)]

    for t in range(test_time):

        shuffled_list = data_list[:]
        random.shuffle(shuffled_list)


        for i in range(1,n+1):

            num=int(i/n*len(shuffled_list))

            err_temp=drp.SR_partition(shuffled_list[:num],epsilon,mean_true)

            #print(abs(pol.x2_cal(shuffled_list,2/epsilon)-result_true)/result_true)

            err_sum_list[i-1]+=err_temp

    return [q/test_time for q in err_sum_list]



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
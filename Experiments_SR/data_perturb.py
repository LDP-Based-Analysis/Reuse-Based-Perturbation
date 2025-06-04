import time

import numpy as np
import time
def ones(true_list):
    a=min(true_list)
    b=max(true_list)
    k=2/(b-a)

    # return -1+(k*np.array(true_list)-a)
    return [-1+k*(i-a) for i in true_list]

def SR(true_list,epsilon):

    Q=1/(np.e**epsilon+1)
    P=1-Q

    k=2/(max(true_list)-min(true_list))
    a=min(true_list)

    mean_ture=sum(true_list)/len(true_list)

    true_list=ones(true_list)

    noisy_list=[]

    for i in true_list:
        if np.random.random()<0.5+(P-Q)/2*i:
            noisy_list.append(1)
        else:
            noisy_list.append(-1)

    mean_estimate=(sum(noisy_list)/(P-Q)/len(true_list)+1)/k+a

    return [abs(mean_estimate-mean_ture)/mean_ture,mean_estimate]



def SR_partition(true_list,epsilon,mean_true):

    Q=1/(np.e**epsilon+1)
    P=1-Q

    k=2/(max(true_list)-min(true_list))
    a=min(true_list)

    true_list=ones(true_list)

    noisy_list=[]

    for i in true_list:
        if np.random.random()<0.5+(P-Q)/2*i:
            noisy_list.append(1)
        else:
            noisy_list.append(-1)

    mean_estimate=(sum(noisy_list)/(P-Q)/len(true_list)+1)/k+a


    #print(mean_estimate,mean_true)

    return abs(mean_estimate-mean_true)/mean_true


def SR_reuse(true_list,epsilon_list):
    mean_ture = sum(true_list) / len(true_list)

    k = 2 / (max(true_list) - min(true_list))
    a = min(true_list)

    true_list = ones(true_list)
    E_x2 = sum([i ** 2 for i in true_list]) / len(true_list)

    noisy_mar = []
    estimate_mar = []
    weight_list = []

    epsilon_max = 0

    for i in range(len(epsilon_list)):

        epsilon = epsilon_list[i]

        Q = 1 / (np.e ** epsilon + 1)
        P = 1 - Q



        noisy_list = []
        for i in true_list:
            if np.random.random() < 0.5 + (P - Q) / 2 * i:
                noisy_list.append(1)
            else:
                noisy_list.append(-1)


        noisy_mar.append(noisy_list)
        estimate_mar.append((sum(noisy_list)/(P-Q)/len(true_list)+1)/k+a)

        weight_list.append((P-Q)**2/(1-(P-Q)**2*E_x2))

        if epsilon > epsilon_max:
            epsilon_max = epsilon
            epsilon_max_list = estimate_mar[-1]

    sum_weight = sum(weight_list)

    estimate_mean = 0
    for j in range(len(epsilon_list)):
        estimate_mean += estimate_mar[j] * weight_list[j] / sum_weight

    return abs(epsilon_max_list-mean_ture)/mean_ture, abs(estimate_mean-mean_ture)/mean_ture

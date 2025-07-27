import numpy as np

def error_cal(true_list,estimate_list):
    return sum([(true_list[i] - estimate_list[i]) ** 2 for i in range(len(true_list))]) / len(true_list)

def UE(true_list,number1,epsilon):
    P = 0.5
    Q = 1 / (np.e ** epsilon + 1)

    number1=sum(true_list)

    noisy_list = [np.random.binomial(i, P) + np.random.binomial(number1 - i, Q) for i in true_list]


    estimate_list = [(i - number1 * Q) / (P - Q) for i in noisy_list]

    return [error_cal(true_list,estimate_list),estimate_list]

def UE_partition(true_list,epsilon):

    number1=sum(true_list)

    P = 0.5
    Q = 1 / (np.e ** epsilon + 1)

    noisy_list = [np.random.binomial(i, P) + np.random.binomial(number1 - i, Q) for i in true_list]

    estimate_list = [(i - number1 * Q) / (P - Q) for i in noisy_list]

    return estimate_list

def UE_reuse(true_list,number1,epsilon_list):

    noisy_mar=[]
    estimate_mar=[]
    weight_list=[]

    epsilon_max=0

    for i in range(len(epsilon_list)):

        epsilon = epsilon_list[i]

        P = 0.5
        Q = 1 / (np.e ** epsilon + 1)

        noisy_mar.append([np.random.binomial(i, P)+np.random.binomial(number1-i,Q)  for i in true_list])
        estimate_mar.append([(j-number1*Q)/(P-Q) for j in noisy_mar[i]])

        weight_list.append((P-Q)**2/Q*(1-Q))

        if epsilon > epsilon_max:
            epsilon_max=epsilon
            epsilon_max_list=estimate_mar[-1]

    sum_weight=sum(weight_list)

    estimate_list = np.zeros(np.array(estimate_mar[0]).shape)
    for j in range(len(epsilon_list)):
        estimate_list += np.array(estimate_mar[j]) * weight_list[j] / sum_weight

    return error_cal(true_list, epsilon_max_list), error_cal(true_list, estimate_list)
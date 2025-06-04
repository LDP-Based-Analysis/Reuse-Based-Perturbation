import numpy as np

def error_cal(true_list,estimate_list):
    return sum([(true_list[i] - estimate_list[i]) ** 2 for i in range(len(true_list))]) / len(true_list)

def kRR(true_list,number1,epsilon):

    k = len(true_list)

    P = np.e ** (epsilon) / (np.e ** epsilon + k - 1)
    Q = (1 - P) / (k - 1)

    noisy_list = [np.random.binomial(i, P) + np.random.binomial(number1 - i, Q) for i in true_list]

    estimate_list = [(i - number1*Q) / (P - Q) for i in noisy_list]

    return [error_cal(true_list,estimate_list),estimate_list]


def kRR_partition(true_list,epsilon):

    k=len(true_list)
    number1=sum(true_list)

    P = np.e ** (epsilon) / (np.e ** epsilon + k - 1)
    Q = (1 - P) / (k - 1)

    noisy_list = [np.random.binomial(i, P) + np.random.binomial(number1 - i, Q) for i in true_list]

    estimate_list = [(i - number1 * Q) / (P - Q) for i in noisy_list]

    return estimate_list


def kRR_reuse(true_list,number1,epsilon_list):
    k = len(true_list)

    noisy_mar=[]
    estimate_mar=[]
    weight_list=[]

    epsilon_max=0

    for i in range(len(epsilon_list)):

        epsilon = epsilon_list[i]

        P = np.e ** (epsilon) / (np.e ** epsilon + k - 1)
        Q = (1 - P) / (k - 1)

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
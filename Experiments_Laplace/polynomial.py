import numpy as np

def x2_cal(degree_list,lamda):

    return sum([(i**2-2*lamda**2-i)/2 for i in degree_list])


def x3_cal(degree_list,lamda):
    return sum([(i**3-6*i*lamda**2-3*(i**2-2*lamda**2)+2*i)/6 for i in degree_list])

def x2_cal_baseline(noisy_mar,epsilon_list):
    v_sum=0
    w_sum=0

    for i in range(len(epsilon_list)):
        v_sum+= x2_cal(noisy_mar[i],2/epsilon_list[i]) * epsilon_list[i]**2
        w_sum+=epsilon_list[i]**2

    return v_sum/w_sum


def x3_cal_baseline(noisy_mar, epsilon_list):
    v_sum = 0
    w_sum = 0

    for i in range(len(epsilon_list)):
        v_sum += x3_cal(noisy_mar[i],2/epsilon_list[i]) * epsilon_list[i] ** 2
        w_sum += epsilon_list[i] ** 2

    return v_sum / w_sum
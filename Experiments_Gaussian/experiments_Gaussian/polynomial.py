import numpy as np


def x2_cal(data_gau,sig):
    return (sum([(i**2-sig**2-i)/2 for i in data_gau]))

def x3_cal(data_gau,sig):
    return (sum([(i**3-3*i*sig**2-3*(i**2-sig**2)+2*i)/6 for i in data_gau]))


def x2_cal_baseline(noisy_mar,sigma_list):
    v_sum = 0
    w_sum = 0

    for i in range(len(sigma_list)):
        v_sum += x2_cal(noisy_mar[i], sigma_list[i]) * 1 / sigma_list[i]**2
        w_sum += 1 / sigma_list[i] ** 2

    return v_sum / w_sum


def x3_cal_baseline(noisy_mar, sigma_list):
    v_sum = 0
    w_sum = 0

    for i in range(len(sigma_list)):
        v_sum += x3_cal(noisy_mar[i], sigma_list[i]) * 1 / sigma_list[i] ** 2
        w_sum += 1 / sigma_list[i] ** 2

    return v_sum / w_sum
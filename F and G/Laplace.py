import random
import numpy as np
from scipy.stats import ks_2samp

import numpy as np
from scipy.stats import rv_continuous

from scipy.stats import wasserstein_distance as was
import time

def Laplace_F(num_noisy,epsilon_1,epsilon_2):  # F function for Laplace
    P=epsilon_2**2/epsilon_1**2

    if random.random()<P:
        return num_noisy
    else:
        return num_noisy+np.random.laplace(0,1/epsilon_2)


def Laplace_G(x_noisy,v,b1, b2):
    x2 = abs(x_noisy)

    F_sum = 1 - (b1 / b2) * np.exp((b1 - b2) * x2 / (b1 * b2))

    U = np.random.uniform(0, 1)

    if U > F_sum:
        return v+x_noisy

    F_0 = (b2 / (2 * (b1 + b2))) * (1 - b1 ** 2 / b2 ** 2)
    # the value of Integral results when 0
    F_x2 = ((b2 / (2 * (b2 - b1))) * (1 - np.exp((b1 - b2) * x2 / (b1 * b2))) + b2 / (2 * (b1 + b2))) * (1 - b1 ** 2 / b2 ** 2)
    # the value of Integral results when x_2


    if U < F_0:  # (-∞, 0)     # the Integral results of probability density function for G of Laplace
        x1 = (b1 * b2 / (b1 + b2)) * np.log(2 * U/(1-b1**2/b2**2) * (b1 + b2) / b2)

    elif U < F_x2:  # (0, x2)
        x1 = (b1 * b2 / (b1 - b2)) * np.log(1 - (2 * (b2 - b1) * (U - F_0)/(1-b1**2/b2**2)) / b2)

    else: #(x2, ∞)
        x1 = b1*b2/(b1+b2) * (2*x2/b2 - np.log(np.e**(x2/b2-x2/b1)-2 * (b1 + b2) / b2 * (U - F_x2)/(1-b1**2/b2**2)))


    if x_noisy >= 0:
        return v+x1
    else:
        return v-x1

def Laplace(num,epsilon_1):
    return (num+np.random.laplace(0,1/epsilon_1))

epsilon_1=2
epsilon_2=1

v=100

length=14000


test_time=100

was_22=0# the meaning of was_22, was_12, p_22, p_12, was_11, was_21, p_11, p_21 can be seen in main.py
was_12=0
p_22=0
p_12=0

was_11=0
was_21=0
p_11=0
p_21=0

time_direct=0# the comparision on running time between direct perturbation, F and G
time_F=0
time_G=0

for i in range(test_time):
    time_direct_1=time.time()
    list_1 = [Laplace(v, epsilon_1) for i in range(length)]
    time_direct_2=time.time()

    list_2 = [Laplace(v, epsilon_2) for i in range(length)]

    time_F_1=time.time()
    list_12 = [Laplace_F(i, epsilon_1, epsilon_2) for i in list_1]
    time_F_2=time.time()

    time_G_1=time.time()
    list_21 = [Laplace_G(i - v, v, 1 / epsilon_1, 1 / epsilon_2) for i in list_2]
    time_G_2=time.time()

    list_10 = [Laplace(v, epsilon_1) for i in range(length)]
    list_20 = [Laplace(v, epsilon_2) for i in range(length)]


    was_22+=was(list_2,list_20)
    was_12+=was(list_2,list_12)

    stat, p_value = ks_2samp(list_2, list_20)
    p_22+=p_value

    stat, p_value = ks_2samp(list_2, list_12)
    p_12+=p_value


    was_11+=was(list_1,list_10)
    was_21+=was(list_1,list_21)

    stat, p_value = ks_2samp(list_1, list_10)
    p_11 += p_value

    stat, p_value = ks_2samp(list_1, list_21)
    p_21 += p_value

    time_direct += (time_direct_2 - time_direct_1)
    time_F += (time_F_2 - time_F_1)
    time_G += (time_G_2 - time_G_1)


print(was_22/test_time,"\t",was_12/test_time)
print(p_22/test_time,"\t",p_12/test_time)

print(was_11/test_time,"\t",was_21/test_time)
print(p_11/test_time,"\t",p_21/test_time)

print(time_direct/test_time,time_F/test_time,time_G/test_time)
# print(was_12/test_time)
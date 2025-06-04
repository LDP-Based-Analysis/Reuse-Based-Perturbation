import random
import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance as was

import numpy as np
from scipy.stats import rv_continuous

import time

def Gaussian_F(num_noisy,s1,s2): #F function of Gaussian

    # sig=(s2**2-s1**2)**0.5

    # return num_noisy+np.random.normal(0,sig)

    return num_noisy+np.random.normal(0,(s2**2-s1**2)**0.5)


def Gaussian_G(x2,sigma1, sigma2): #G function of Gaussian
    mu_x1 = (sigma1 ** 2 / sigma2 ** 2) * x2
    sigma_x1 = np.sqrt(sigma1 ** 2 * (sigma2 ** 2 - sigma1 ** 2) / sigma2 ** 2)

    return np.random.normal(mu_x1, sigma_x1)

delta=10**(-5)
e1=2
e2=1


s1=(2*np.log(1.25/delta))**0.5/e1 # \sigma corresponds to e1
s2=(2*np.log(1.25/delta))**0.5/e2 # \sigma corresponds to e2

length=10000  # the value of k

test_time=100  # repeat how many times

was_22=0 # the meaning of was_22, was_12, p_22, p_12, was_11, was_21, p_11, p_21 can be seen in main.py
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
    time_direct_1 = time.time()
    list_1=[np.random.normal(0,s1) for i in range(length)]
    time_direct_2 = time.time()# the running time of direct perturbation

    list_2=[np.random.normal(0,s2) for i in range(length)]

    time_F_1=time.time()
    list_12=[Gaussian_F(i,s1,s2) for i in list_1]
    time_F_2=time.time()# the running time of F

    time_G_1=time.time()
    list_21=[Gaussian_G(i,s1,s2) for i in list_2]
    time_G_2=time.time()# the running time of G

    list_10=[np.random.normal(0,s1) for i in range(length)]
    list_20=[np.random.normal(0,s2) for i in range(length)]


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

    time_direct+=(time_direct_2-time_direct_1)
    time_F+=(time_F_2-time_F_1)
    time_G+=(time_G_2-time_G_1)


print(was_22/test_time,"\t",was_12/test_time)
print(p_22/test_time,"\t",p_12/test_time)

print(was_11/test_time,"\t",was_21/test_time)
print(p_11/test_time,"\t",p_21/test_time)

print(time_direct/test_time,time_F/test_time,time_G/test_time)

# print(was_22/test_time)

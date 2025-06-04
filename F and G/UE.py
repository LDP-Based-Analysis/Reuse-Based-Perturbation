import random
import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance as was
import time

def OUE_F(num_noisy,epsilon_1,epsilon_2): #F function for UE, select OUE as example

    p1=1/2
    q1=1/(np.e**epsilon_1+1)

    p2=1/2
    q2=1/(np.e**epsilon_2+1)

    f1=(p1*q2-p2*q1)/(p1-q1)
    f2=(p1*q2-p2*q1+p2-q2)/(p1-q1)

    if num_noisy==0:
        if random.random()<f1:
            return 1
        else:
            return 0
    else:
        if random.random()<f2:
            return 1
        else:
            return 0

def OUE_G(num_true,num_noisy,epsilon_1,epsilon_2):#G function for UE, select OUE as example
    p1 = 1 / 2
    q1 = 1 / (np.e ** epsilon_1 + 1)

    p2 = 1 / 2
    q2 = 1 / (np.e ** epsilon_2 + 1)

    g1 = ((1 - q1) * (p1 - p1 * q2 + p2 * q1 - q1)) / ((p1 - q1) * (1 - q2))
    g2 = ((1 - p1) * (p1 - p1 * q2 + p2 * q1 - q1)) / ((p1 - q1) * (1 - p2))
    g3 = (q1 * (p1 * q2 - p2 * q1 + p2 - q2)) / (q2 * (p1 - q1))
    g4 = (p1 * (p1 * q2 - p2 * q1 + p2 - q2)) / (p2 * (p1 - q1))

    if num_noisy==0 and num_true==0:
        if random.random()<g1:
            return 0
        else:
            return 1
    if num_noisy==0 and num_true==1:
        if random.random()<g2:
            return 0
        else:
            return 1
    if num_noisy==1 and num_true==0:
        if random.random()<g3:
            return 1
        else:
            return 0
    else:
        if random.random()<g4:
            return 1
        else:
            return 0

def OUE(num_true,epsilon_1):#OUE

    P=1/2
    Q=1/(np.e**epsilon_1+1)

    if num_true==1:
        if random.random()<P:
            return num_true
        else:
            return 1-num_true
    else:
        if random.random()<Q:
            return 1-num_true
        else:
            return num_true


e1=2
e2=1

num_true_list=[1 for i in range(6000)]+[0 for i in range(4000)]# can be changed to any other

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
    list_1 = [OUE(i, e1) for i in num_true_list]
    time_direct_2=time.time()

    list_2 = [OUE(i, e2) for i in num_true_list]

    time_F_1=time.time()
    list_12 = [OUE_F(list_1[i], e1, e2) for i in range(len(num_true_list))]
    time_F_2=time.time()

    time_G_1=time.time()
    list_21 = [OUE_G(num_true_list[i], list_2[i], e1, e2) for i in range(len(num_true_list))]
    time_G_2=time.time()

    list_10 = [OUE(i, e1) for i in num_true_list]
    list_20 = [OUE(i, e2) for i in num_true_list]


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
#print(was_12/test_time)
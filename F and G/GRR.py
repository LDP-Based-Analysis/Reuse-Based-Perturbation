import random
import numpy as np

from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance as was
import time

def GRR_F(num_noisy,num_set,epsilon_1,epsilon_2):  # F function for GRR
    k=len(num_set)

    P0=np.e**(epsilon_1)/(np.e**(epsilon_1)+k-1)
    P2=np.e**(epsilon_2)/(np.e**(epsilon_2)+k-1)

    Q1=((k-1)*P2+P0-1)/(k*P0-1)
    Q_2=(1-Q1)/(k-1)

    if random.random()<Q1-Q_2:
        return num_noisy
    else:
        return random.choice(num_set)


def GRR_G(num_noisy,num_true,num_set,essilon_1,epsilon_2): # G function for GRR
    eps1=essilon_1
    eps2=epsilon_2

    d=len(num_set)

    g1 = (np.exp(eps1) / np.exp(eps2)) * (
                (np.exp(eps1) * (np.exp(eps2) - 1) / (np.exp(eps1) - 1) + d - 1) / (np.exp(eps1) + d - 1))
    g2 = (np.exp(eps1) * (np.exp(eps2) - 1) / (np.exp(eps1) - 1) + d - 1) / (np.exp(eps1) + d - 1)

    if num_noisy==num_true:
        if random.random()<g1-(1-g1)/(d-1):
            return num_noisy
        else:
            return random.choice(num_set)
    else:
        if random.random()<g2-(1-g2)/(d-1):
            return num_noisy
        else:
            return random.choice(num_set)



def GRR(num_true,num_set,epsilon_1): #GRR
    k=len(num_set)

    P=(np.e**(epsilon_1)-1)/(np.e**epsilon_1+k-1)

    if random.random()<P:
        return num_true
    else:
        return random.choice(num_set)

num_set=[1,2,3]  # can be changed to any other

e1=0.5 # e1 should satisfy that e1 < ln((d-1)/(d-2))
e2=0.1

length=10000

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
    list_1 = [GRR(2, num_set, e1) for i in range(length)]
    time_direct_2=time.time()

    list_2 = [GRR(2, num_set, e2) for i in range(length)]

    time_F_1=time.time()
    list_12 = [GRR_F(i, num_set, e1, e2) for i in list_1]
    time_F_2=time.time()

    time_G_1=time.time()
    list_21 = [GRR_G(i, 2, num_set, e1, e2) for i in list_2]
    time_G_2=time.time()

    list_10 = [GRR(2, num_set, e1) for i in range(length)]
    list_20 = [GRR(2, num_set, e2) for i in range(length)]


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

# print(was_22/test_time)
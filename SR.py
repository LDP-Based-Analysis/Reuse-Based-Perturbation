import random
import numpy as np
from scipy.stats import ks_2samp

from scipy.stats import wasserstein_distance as was


def SR_geq(num_noisy,epsilon_1,epsilon_2):
    q1 = 1 / (np.e ** epsilon_1 + 1)
    p1 = 1 - q1

    q2 = 1 / (np.e ** epsilon_2 + 1)
    p2 = 1 - q2

    f1=(p1*q2-p2*q1)/(p1-q1)
    f2=(p1*q2-p2*q1+p2-q2)/(p1-q1)


    if num_noisy==1:
        if random.random()<f2:
            return 1
        else:
            return -1
    else:
        if random.random()<f2:
            return -1
        else:
            return 1


def SR_leq(num_noisy,num_true,epsilon_1,epsilon_2):
    q1 = 1 / (np.e ** epsilon_1 + 1)
    p1 = 1 - q1

    q2 = 1 / (np.e ** epsilon_2 + 1)
    p2 = 1 - q2

    P1=0.5+0.5*(p1-q1)*num_true
    P2=0.5+0.5*(p2-q2)*num_true


    g1 = (P1 * (P1 + P2 - 1)) / (P2 * (2 * P1 - 1))
    g2 = ((1 - P1) * (P1 + P2 - 1)) / ((1 - P2) * (2 * P1 - 1))

    if num_noisy * num_true>=0:
        if random.random()<g1:
            return num_noisy
        else:
            return 0-num_noisy
    else:
        if random.random()<g2:
            return num_noisy
        else:
            return 0-num_noisy



def SR_reuse(num_noisy,num_true,epsilon_1,epsilon_2):
    if epsilon_1>epsilon_2:
        return SR_geq(num_noisy,epsilon_1,epsilon_2)
    if epsilon_1<epsilon_2:
        return SR_leq(num_noisy,num_true,epsilon_2,epsilon_1)
    if epsilon_1==epsilon_2:
        return num_noisy


def SR(num_true,epsilon_1):

    Q=1/(np.e**epsilon_1+1)
    P=1-Q

    if random.random()<(0.5+0.5*(P-Q)*num_true):
        return 1
    else:
        return -1




epsilon_list=[0.5,0.7,0.9,1.1,1.3,1.5]

num_true_list=[1 for i in range(6000)]+[0 for i in range(4000)]

e1=7
e2=2

v=0.5

length=10000

test_time=100

was_22=0
was_12=0
p_22=0
p_12=0

was_11=0
was_21=0
p_11=0
p_21=0

for i in range(test_time):
    list_1 = [SR(v, e1) for i in range(length)]
    list_2 = [SR(v, e2) for i in range(length)]


    list_12 = [SR_reuse(list_1[i], v, e1, e2) for i in range(length)]
    list_21 = [SR_reuse(list_2[i], v, e2, e1) for i in range(length)]

    list_10 = [SR(v, e1) for i in range(length)]
    list_20 = [SR(v, e2) for i in range(length)]


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


# print(was_22/test_time,"\t",was_12/test_time)
# print(p_22/test_time,"\t",p_12/test_time)
#
# print(was_11/test_time,"\t",was_21/test_time)
# print(p_11/test_time,"\t",p_21/test_time)

print(was_12/test_time)
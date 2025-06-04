import cvxpy as cp
import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance as was
import time

def sw_matrix(d, b, e): # the matrix form of SW
    p1 = np.e ** (e) / (np.e ** e * (2 * b + 1) + d - 1)
    q1 = 1 / (np.e ** e * (2 * b + 1) + d - 1)

    m = np.zeros((d, d + 2 * b))

    for i in range(d):
        for j in range(d + 2 * b):
            if (abs(j - (i + b)) <= b):
                m[i, j] = p1
            else:
                m[i, j] = q1

    return m


def sw_solve(M_1, M_2): # the solution for M_F
    n1, m1 = M_1.shape
    n2, m2 = M_2.shape

    # the shape of M_F
    B = cp.Variable((m1, m2))

    #
    constraints = [
        B >= 10**(-5), # positive
        cp.sum(B, axis=1) == 1  # sum=1
    ]# two types of constraints

    # minimize ||A @ B - C||_F^2
    objective = cp.Minimize(cp.sum_squares(M_1 @ B - M_2)) # the first constraint

    # solve
    problem = cp.Problem(objective, constraints)
    result = problem.solve()

    # 输出结果
    if problem.status in ['optimal', 'optimal_inaccurate']:
        B_optimal = B.value

    else:
        print("未能找到满足条件的解。问题状态:", problem.status)

    return (B_optimal)


def sw_F(new_list, M_3, d, b): # F function for SW
    end_list = []

    for x in new_list:
        list_p = M_3[x + 1]

        list_p = np.array(list_p) / sum(list_p)

        noisy_values = [(i - b + 1) for i in range(d + 2 * b)]
        end_list.append(np.random.choice(noisy_values, p=list_p))
    return end_list

def sw_list(x, d, b, e, n): # SW perturbation to generate random
    M = sw_matrix(d, b, e)

    list_p = M[x - 1]

    noisy_values = [(i - b + 1) for i in range(d + 2 * b)]

    new_list = [np.random.choice(noisy_values, p=list_p) for t in range(n)]

    return new_list


d=10
b1=2# b corresponds to e1

b2=3# b corresponds to e2

e1=2
e2=1

v=2# can be changed to any other
length=10000


M_1=(sw_matrix(2,2,e1))
M_2=(sw_matrix(2,2,e2))
M_12=sw_solve(M_1,M_2)

print(M_12)

M_1=(sw_matrix(d,b1,e1))
M_2=(sw_matrix(d,b2,e2))
M_12=sw_solve(M_1,M_2)

print(M_12)

test_time=100
was_22=0# the meaning of was_22, was_12, p_22, p_12, was_11, was_21, p_11, p_21 can be seen in main.py
was_12=0
p_22=0
p_12=0

was_11=0
was_21=0
p_11=0
p_21=0

time_direct=0# the comparision on running time between direct perturbation, F
time_F=0

for i in range(test_time):
    time_direct_1=time.time()
    list_1 = sw_list(v, d, b1, e1, length)
    time_direct_2=time.time()

    list_2 = sw_list(v, d, b2, e2, length)

    time_F_1=time.time()
    list_12 = sw_F(list_1, M_12, d, b2)
    time_F_2=time.time()

    list_20 = sw_list(v, d, b2, e2, length)

    was_22+=was(list_2,list_20)
    was_12+=was(list_2,list_12)

    stat, p_value = ks_2samp(list_2, list_20)
    p_22+=p_value

    stat, p_value = ks_2samp(list_2, list_12)
    p_12+=p_value

    time_direct += (time_direct_2 - time_direct_1)
    time_F += (time_F_2 - time_F_1)


print(was_22/test_time,"\t",was_12/test_time)
print(p_22/test_time,"\t",p_12/test_time)


print(time_direct/test_time,time_F/test_time)
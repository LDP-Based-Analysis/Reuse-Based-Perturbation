import numpy as np
from scipy.stats import wasserstein_distance as was
import cvxpy as cp
from scipy.stats import ks_2samp

import time


def matrix_OUE(e):
    m=np.array([[np.e**e/(np.e**e+1),1/(np.e**e+1)],[0.5,0.5]])

    return m

def matrix_GRR(d,e):
    p1 = np.e**(e) / (np.e**e + d - 1)
    q1 = 1 / (np.e**e + d - 1)

    m= np.zeros((d,d))

    for i in range(d):
        for j in range(d):
            if i==j:
                m[i,j]=p1
            else:
                m[i,j]=q1
    return m


def matrix_SR(d,e):

    p=np.e**e/(np.e**e+1)
    q=1-p

    m = np.zeros((d,2))

    for x in range(1,d+1):
        B = -1 + 2/(d-1) * (x-1)

        m[x-1,0] = 0.5 - (p-q)/2*B
        m[x-1,1] = 1 - m[x-1,0]

    return m


def matrix_SW(d, b, e): # the matrix form of SW
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


def strict_M_F(M1, M2):
    # 计算秩
    rank_M1 = np.linalg.matrix_rank(M1)
    rank_M1_M2 = np.linalg.matrix_rank(np.hstack((M1, M2)))  # 扩展 M1 和 M2 计算秩


    # 条件 1: rank(M1) == rank(M1 : M2)
    if rank_M1 == rank_M1_M2:

        M_T, residuals, rank, s = np.linalg.lstsq(M1, M2, rcond=None)

        non_negative_elements = np.all(M_T >= 0)

        if non_negative_elements:
            return M_T
        else:
            return None

    else:
        #print("No solutions exist.")
        return None


def releaxed_M_F(M_1, M_2):
    n1, m1 = M_1.shape
    n2, m2 = M_2.shape

    # the shape of M_F
    B = cp.Variable((m1, m2))

    #
    constraints = [
        B >= 10 ** (-5),  # positive
        cp.sum(B, axis=1) == 1  # sum=1
    ]  # two types of constraints

    # minimize ||A @ B - C||_F^2
    objective = cp.Minimize(cp.sum_squares(M_1 @ B - M_2))  # the first constraint

    # solve
    problem = cp.Problem(objective, constraints)
    result = problem.solve()

    #print((abs(M_1@B.value-M_2)))

    # 输出结果
    if problem.status in ['optimal', 'optimal_inaccurate'] and np.max(abs(M_1@B.value-M_2))<= 10**(-3):
        B_optimal = B.value

    else:
        print("未能找到满足条件的解。问题状态:", problem.status)
        return None

    return (B_optimal)

def G_Matrix(M1,M2,M_transition,x_raw,x_noisy,noise_list):

    index=noise_list.index(x_noisy)

    pro_list=[]
    for i in range(M1.shape[1]):
        pro_list.append(M1[x_raw-1,i]*M_transition[i,index])

    total_sum = sum(pro_list)

    # 进行归一化
    normalized_probabilities = [p / total_sum for p in pro_list]

    return normalized_probabilities


def noise_list(pro_list,repeat_time,domain_list): # SW perturbation to generate random

    new_list = [np.random.choice(domain_list, p=pro_list) for t in range(repeat_time)]

    return new_list

d=10
b1=2
b2=2

raw_data=2

e1=3
e2=1

repeat_time=10000

M1=matrix_GRR(d,e1)#set the parametter M1 as matrix_UE(e1) or matrix_GRR(d,e1) or matrix_SR(d,e1) or matrix_SW(d,b1,e1)
M2=matrix_SR(d,e2)# similar to the M2

M_T_strict=strict_M_F(M1,M2)

M_T_relax=releaxed_M_F(M1,M2)

if M_T_strict is not None:
    M_T=M_T_strict
elif M_T_relax is not None:
    M_T=M_T_relax
else:
    print("M_T does not exist")

# #UE to UE
# input_domin=[0,1]#UE
# output_domain_e1=[0,1]#OUE
# output_domain_e2=[0,1]#OUE


#SR to SR
# input_domain=[i for i in range(1,d+1)]#GRR, SR, SW
# output_domain_e1=[-1,1]#SR
# output_domain_e2=[-1,1]#SR
#
#
#SW to SW
# input_domain=[i for i in range(1,d+1)]#GRR, SR, SW
# output_domain_e1=[(i - b1 + 1) for i in range(d + 2 * b1)]#SW
# output_domain_e2=[(i - b2 + 1) for i in range(d + 2 * b2)]#SW
#
#
# #GRR to SW
# input_domain=[i for i in range(1,d+1)]#GRR, SR, SW
# output_domain_e1=[i for i in range(1,d+1)]##GRR
# output_domain_e2=[(i - b2 + 1) for i in range(d + 2 * b2)]#SW
#
#
# #GRR to SR
input_domain=[i for i in range(1,d+1)]#GRR, SR, SW
output_domain_e1=[i for i in range(1,d+1)]##GRR
output_domain_e2=[-1,1]#SR



pro_list_e1=M1[raw_data-1]
pro_list_e2=M2[raw_data-1]




test_time=10

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
time_G=0

for i in range(test_time):
    time_direct_1=time.time()
    noisy_list_e1 = noise_list(pro_list_e1, repeat_time, output_domain_e1)
    time_direct_2=time.time()

    time_direct+=(time_direct_2-time_direct_1)

    noisy_list_e2 = noise_list(pro_list_e2, repeat_time, output_domain_e2)

    time_F_1=time.time()
    x_F_newlist=[noise_list(M_T[output_domain_e1.index(i)],1,output_domain_e2)[0] for i in noisy_list_e1]
    time_F_2=time.time()
    stat, p_value = ks_2samp(noisy_list_e2, x_F_newlist)
    # print("e1 and F:\t",was(noisy_list_e2,x_F_newlist),"\t",p_value)

    was_12+=was(noisy_list_e2,x_F_newlist)
    p_12+=p_value
    time_F+=(time_F_2-time_F_1)


    time_G_1=time.time()
    x_G_newlist=[noise_list(G_Matrix(M1,M2,M_T,raw_data,i,output_domain_e2),1,output_domain_e1)[0] for i in noisy_list_e2]
    time_G_2=time.time()
    stat, p_value = ks_2samp(noisy_list_e1, x_G_newlist)

    was_21+=was(noisy_list_e1,x_G_newlist)
    p_21+=p_value

    time_G+=(time_G_2-time_G_1)


#begin independent perturbation

    noisy_list_e2_new = noise_list(pro_list_e2, repeat_time, output_domain_e2)
    stat, p_value = ks_2samp(noisy_list_e2, noisy_list_e2_new)
    was_22+=was(noisy_list_e2, noisy_list_e2_new)
    p_22+=p_value

    noisy_list_e1_new = noise_list(pro_list_e1, repeat_time, output_domain_e1)
    stat, p_value = ks_2samp(noisy_list_e1, noisy_list_e1_new)
    was_11+=was(noisy_list_e1,noisy_list_e1_new)
    p_11+=p_value


print(was_22/test_time,"\t",was_12/test_time) # validation for F
print(p_22/test_time,"\t",p_12/test_time) # validation for F

print(was_11/test_time,"\t",was_21/test_time) # validation for G
print(p_11/test_time,"\t",p_21/test_time) # validation for G


print(time_direct/test_time,time_F/test_time,time_G/test_time)# running time comparison between direct perturbation, F and G
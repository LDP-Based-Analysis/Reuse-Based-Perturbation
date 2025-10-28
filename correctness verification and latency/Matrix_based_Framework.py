import numpy as np
from scipy.stats import wasserstein_distance as was
import cvxpy as cp
from scipy.stats import ks_2samp

import os
import psutil

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


def relaxed_column(M1, M2):          ### the alternative method solving for approximate M_F
    n, m = M1.shape
    _, p = M2.shape
    B = np.zeros((m, p))

    time_start = time.time()
    for j in range(p):

        b = cp.Variable(m)
        objective = cp.Minimize(cp.sum_squares(M1 @ b - M2[:, j]))
        constraints = [b >= 10**(-5)]
        prob = cp.Problem(objective, constraints)
        prob.solve()#solver=cp.OSQP)
        B[:, j] = b.value
        time_end = time.time()

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"RSS: {mem_info.rss / 1024 ** 2:.2f} MB")  # 常用物理内存占用

        print(j,time_end-time_start)


    # row_sums = B.sum(axis=1, keepdims=True)
    # B = B / row_sums

    B=projection_on_simplex_rows(B)

    print(np.max(abs(M1@B-M2)))

    # if np.max(abs(M1@B-M2))<= 10**(-3):
    #     return B
    #
    # else:
    #     print("未能找到满足条件的解。问题状态:")
    #     return None

    return B


def simplex_proj(v):
    """投影向量 v 到概率单纯形 {x| x>=0, sum(x)=1}"""
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)
    rho_arr = u - (sv - 1) / (np.arange(len(u)) + 1)
    rho = np.where(rho_arr > 0)[0][-1]
    theta = (sv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w


def projection_on_simplex_rows(B):
    """对矩阵B每一行进行单纯形投影"""
    B_proj = np.zeros_like(B)
    for i in range(B.shape[0]):
        B_proj[i, :] = simplex_proj(B[i, :])
    return B_proj


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


def max_ratio_per_column(matrix):
    ratios = []
    for col in range(matrix.shape[1]):
        col_vals = matrix[:, col]
        # 排除0和负数时根据实际需求定义，若允许负值，可能需要调整
        # 这里假设所有元素均为正数
        min_val = np.min(col_vals[col_vals > 0])  # 最小正值
        max_val = np.max(col_vals)
        ratio = max_val / min_val if min_val != 0 else np.inf
        ratios.append(ratio)
    return np.array(ratios)


d=10
b1=2
b2=2

raw_data=2

e1=3
e2=1

repeat_time=10000

M1=matrix_SW(d,b1,e1)#set the parametter M1 as matrix_UE(e1) or matrix_GRR(d,e1) or matrix_SR(d,e1) or matrix_SW(d,b1,e1)
M2=matrix_SW(d,b2,e2)# similar to the M2

#M1=matrix_GRR(d,e1)# GRR to SR
#M2=matrix_SR(d,e2)

# M1=matrix_SR(d,3)
# M2=matrix_SR(d,1)

# M1=matrix_OUE(e1) ##UE to UE
# M2=matrix_OUE(e2)



time_1=time.time()

M_T_strict=strict_M_F(M1,M2)

#print(M_T_strict)

if M_T_strict is not None:
    M_T=M_T_strict
    #print("Exact")
else:
    M_T_relax = releaxed_M_F(M1,M2)
    #M_T_relax = relaxed_column(M1,M2)##the alternative method solving for M_F

    if M_T_relax is not None:
        M_T=M_T_relax
        #print("Approx")
    else:
        print("M_T does not exist")


time_2=time.time()

print("time:",time_2-time_1)


num=(max(max_ratio_per_column(M1@M_T)))

print(abs(np.log(num)-e2))



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
input_domain=[i for i in range(1,d+1)]#GRR, SR, SW
output_domain_e1=[(i - b1 + 1) for i in range(d + 2 * b1)]#SW
output_domain_e2=[(i - b2 + 1) for i in range(d + 2 * b2)]#SW
#
#
# #GRR to SW
# input_domain=[i for i in range(1,d+1)]#GRR, SR, SW
# output_domain_e1=[i for i in range(1,d+1)]##GRR
# output_domain_e2=[(i - b2 + 1) for i in range(d + 2 * b2)]#SW
#
#
# #GRR to SR
# input_domain=[i for i in range(1,d+1)]#GRR, SR, SW
# output_domain_e1=[i for i in range(1,d+1)]##GRR
# output_domain_e2=[-1,1]#SR



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
import pandas as pd
import data_perturb as dp
import test_code as test

epsilon_list=[3,1]

# 假设 .data 文件是以逗号分隔的
data = pd.read_csv('dataset/adult.data', header=None)
num1=45222

#i  =   0      ,    1    ,          2    ,  3    ,      4    ,  5
#c  =   17 90,      13492 1490400,  1 16,   0 99999,    0 4356, 1 99

#2,3,4,5;education-num,capital-gain,capital-loss,hours-per-week
column = 3


true_list=list(data[column])

# # Experiments related to, various e_1
# print("Experiment 1 begin: various e_1")
# epsilon_sum_list=[0.1,0.3,0.5,0.7,0.9,1.1,1.3]
# epsilon_tuple=[[epsilon_sum*2/3,epsilon_sum/3] for epsilon_sum in epsilon_sum_list]
#
# for epsilon_list in epsilon_tuple:
#     test.SR_test(true_list,epsilon_list,100)
# #the output correspond to naive select larger privacy budget, baseline, reuse-based perturbation respectively
#
#
# # Experiments related to, various e_2/e_1
# print("Experiment 2 begin: various e_2/e_1")
# alpha_list=[0.5,0.7,0.9]
# epsilon_sum=1
# epsilon_tuple=[[epsilon_sum*i,epsilon_sum*(1-i)] for i in alpha_list]
#
# for epsilon_list in epsilon_tuple:
#     test.SR_test(true_list,epsilon_list,100)
# #the output correspond to naive select larger privacy budget, baseline, reuse-based perturbation respectively



# Experiments related to, various the number of servers k
print("Experiment 3 begin: various server count k")
epsilon=2
server_count=7
data=test.k_test(true_list,epsilon,server_count,test_time=20)
print("\t".join(map(str, data)))
#the output correspond to k=1,...,k=7, respectively
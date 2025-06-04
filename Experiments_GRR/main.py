import data_read_perturb as drp
import pandas as pd
import test_code as test

# 假设 .data 文件是以逗号分隔的
data = pd.read_csv('dataset/adult.data', header=None)
num1=45222

#i  =   0,  1,  2,  3,  4,  5,  6,  7,  8
#c  =   7,  16, 7,  14, 6,  5,  2,  41, 2
column = 2


true_list=list(dict(data[column].value_counts()).values())

# Experiments related to, various e_1
print("Experiment 1 begin: various e_1")
epsilon_sum_list=[0.1,0.4,0.7,1,1.3,1.6,1.9]
epsilon_tuple=[[epsilon_sum*2/3,epsilon_sum/3] for epsilon_sum in epsilon_sum_list]

for epsilon_list in epsilon_tuple:
    test.kRR_test(true_list,num1,epsilon_list,test_time=100)
#the output correspond to naive select larger privacy budget, baseline, reuse-based perturbation respectively


# Experiments related to, various e_2/e_1
print("Experiment 2 begin: various e_2/e_1")
alpha_list=[0.5,0.6,0.7,0.8,0.9]
epsilon_sum=2
epsilon_tuple=[[epsilon_sum*i,epsilon_sum*(1-i)] for i in alpha_list]

for epsilon_list in epsilon_tuple:
    test.kRR_test(true_list,num1,epsilon_list,100)
#the output correspond to naive select larger privacy budget, baseline, reuse-based perturbation respectively


# Experiments related to, various the number of servers k
print("Experiment 3 begin: various server count k")
epsilon=0.1
data=test.k_test(true_list,epsilon,7,100)
print("\t".join(map(str, data)))
#the output correspond to k=1,...,k=7, respectively
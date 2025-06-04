import data_read
import numpy as np
import polynomial as pol

import test_code as test

str1="dataset/data_astro.txt"
str2="dataset/data_enron.txt"
str3="dataset/data_facebook_attributes.txt"
str4="dataset/data_synthetic.txt" #zipf distribution, parameter=1.5, num=1_000_000,value \in [1,10_000]


data_list=data_read.data_read(str4)
delta_list=[0.001/len(data_list),0.001/len(data_list)]


# Experiments related to, various e_1

# print("Experiment 1 begin: various e_1")
# epsilon_sum_list=[0.1,0.3,0.5,0.7,0.9,1.1,1.3]
# epsilon_tuple=[[epsilon_sum*2/3,epsilon_sum/3] for epsilon_sum in epsilon_sum_list]
# for epsilon_list in epsilon_tuple:
#     data = test.x_2star_test(data_list,epsilon_list,delta_list,test_time=20)
#     print("e_1:",sum(epsilon_list),"\t","\t".join(map(str, data)))



# Experiments related to, various e_2/e_1

# print("Experiment 2 begin: various e_2/e_1")
# alpha_list=[0.5,0.7,0.9]
# epsilon_sum=1
# epsilon_tuple=[[epsilon_sum*i,epsilon_sum*(1-i)] for i in alpha_list]
#
#
# for epsilon_list in epsilon_tuple:
#     data = test.x_2star_test(data_list,epsilon_list,delta_list,test_time=20)
#     print("e_2/e_1:",epsilon_list[0]/sum(epsilon_list),"\t".join(map(str, data)))



# Experiments related to, various the number of servers k
print("Experiment 3 begin: various server count k")
epsilon=0.1
delta=0.001/len(data_list)

server_count=7

data=test.k_test(data_list,epsilon,delta,server_count,test_time=10)
print("\t".join(map(str, data)))

#the output correspond to k=1,...,k=7, respectively
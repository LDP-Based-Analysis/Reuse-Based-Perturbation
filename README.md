The experiments related to Q1 and Q3 that illustrates F and G can generate correct distribution and running efficiency can be conducted by

run correctness verification and latency/Matrix_based_framework.py

The detailed meaning of the parameters can be found in the code annotation, which enables the construction of $F$ and $G$ for transitions from UE to UE, SR to SR, SW to SW, GRR to SW, and GRR to SR. It outputs the relevant results for Q1 and Q3.

The experiments related to Q2 that illustrates F and G can enhance utility of UE and SR in different scenarios can be conducted by 

run utility/SR/main.py

run utility/UE/main.py

The detailed meaning of the parameters can be found in the code annotation, which conduct experiments related to various e_1, various e_2/e_1, and various server count k 

the dataset can be selected by the column/stri

UE, column=6, 8, 0, and 2 denote sex, income, workclass, and marital, respectively

SR, column=2, 5, 4, and 3 denote education, gain, loss, and hour, respectively

Requirements

cvxpy==1.6.1

numpy==2.2.0

scipy==1.14.1

pandas==2.2.3

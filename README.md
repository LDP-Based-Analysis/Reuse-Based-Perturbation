Directory: F and G

Gaussian.py, GRR.py, Laplace.py, SR.py, SW.py, and UE.py are the implementations of the **F** function and **G** function for the six mechanisms, respectively.

The experiments related to the **Table that illustrates F and G can generate correct distribution** and **running efficiency** can be conducted by

run Gaussian.py/GRR.py/Laplace.py/SR.py/SW.py/UE.py

Detailed explanation can be seen in the F and G/main.py

Directory: Experiments_Gaussian/GRR/Laplace/SR/UE

The experiments related to **various e_1**, **various e_2/e_1**, and **various server count k** can be conducted for Gaussian/GRR/Laplace/SR/UE by

run main.py

the dataset can be selected by the column/str**i**

UE and GRR, column=6, 8, 0, and 2 denote sex, income, workclass, and marital, respectively

Laplace and Gaussian, str1, str2, str3, and str4 denote Astro, Enron, Facebook, and synthetic, respectively

SR, column=2, 5, 4, and 3 denote education, gain, loss, and hour, respectively

Requirements

cvxpy==1.6.1
numpy==2.2.0
scipy==1.14.1
pandas==2.2.3

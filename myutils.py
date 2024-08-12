
""" Utility Functions
Utility Functions
This module contains utility functions for simulating OTC prices.
Functions:
- get_OTC_path(T_training, np_seed, T_testing, nmax): Generates OTC prices for later simulation.
    - T_training (int): Training period.
    - np_seed (int): Seed for numpy random number generator.
    - T_testing (int): Testing period.
    - nmax (int): Number of paths.
    - p_c (ndarray): Contract price.
    - p_2 (ndarray): Open market price for regular-weight hogs.
    - p_1 (ndarray): Open market price for under-weight hogs.
    - c_h (ndarray): Holding cost.
    - c_p (ndarray): Penalty cost.
    - W1 (ndarray): Under-weight hogs quantity.
    - W2 (ndarray): Regular-weight hogs quantity.
 """


#################################################################### Simulate OTC prices pseudo code#######################################
Function get_OTC_path(T_training, np_seed, T_testing, nmax):
    Initialize constants:
        q = 98107
        alpha = 0.154
        beta = 0.02
        Twarmup = 101
        start_training = 10
        Tmax = T_training + start_training + T_testing

    Initialize W0_factor matrix
    Initialize W0_OTC matrix
    
    Create zero matrices W1 and W2 of shape (nmax, Tmax)

    Define bimodal function:
        - Generates a bimodal distribution of random numbers using two normal distributions
        - Replace any negative values with 0
        - Return the generated distribution

    Seed the numpy random number generator with np_seed
    
    For each T in range (Tmax-1, Tmax):
        Initialize zero matrices:
            pi_o, pi_m, pi_f, p_c, p_2, p_1, c_h, c_p
        Initialize temporary matrices tempPi_O, tempPi_M, tempPi_F

        For each n in range (nmax):
            Set gen_cont to True
            While gen_cont is True:
                Generate bimodal distribution w1_gen for T+1 periods
                Generate normal distribution w2_gen for T+1 periods
                If sum of w1_gen and w2_gen for all T+1 periods <= 150000:
                    Assign w1_gen to W1[n,:]
                    Assign w2_gen to W2[n,:]
                    Set gen_cont to False
            
            Generate factor market prices (Pi_O, Pi_M, Pi_F) during warmup:
                For each t in range (2, length of u_O):
                    Update Pi_O, Pi_M, Pi_F based on previous values and random noise u_O, u_M, u_F
                
                Replace any negative values in Pi_O, Pi_M, Pi_F with 0
            
            Generate OTC prices during warmup:
                Generate random noise (w_C, w_1, w_2, w_H, w_P)
                Initialize zero matrices P_C, P_1, P_2, C_H, C_P
                For each t in range (53, length of w_C):
                    Update P_C, P_1, P_2, C_H, C_P based on previous values, factor prices, and random noise
                
                Replace any negative values in P_C, P_1, P_2, C_H, C_P with 0
            
            Simulate prices for the first period:
                Assign the last warmup prices to pi_o, pi_m, pi_f, p_c, p_2, p_1, c_h, c_p for the first period
            
            Generate forecast for subsequent periods:
                Initialize W_factor matrix
                Generate forecast for factor prices tempPi_O, tempPi_M, tempPi_F
                Replace any negative values in tempPi_O, tempPi_M, tempPi_F with 0
            
            For each i in range (1, T):
                Generate random noise inno_Pi_O, inno_Pi_M, inno_Pi_F
                Generate factor prices (pi_o, pi_m, pi_f) based on previous values and noise
                
                Replace any negative values in pi_o, pi_m, pi_f with 0
                
                Generate forecast for OTC prices p_c, p_2, p_1, c_h, c_p based on previous values, factor prices, and random noise
                
                Replace any negative values in p_c, p_2, p_1, c_h, c_p with 0

            For the terminal period (T):
                Generate final factor prices (pi_o, pi_m, pi_f)
                Replace any negative values in pi_o, pi_m, pi_f with 0
                
                Generate final forecast for OTC prices p_c, p_2, p_1, c_h, c_p
                
                Replace any negative values in p_c, p_2, p_1, c_h, c_p with 0
    
    For each T in range (T_training-1, T_training):
        Initialize Vo as a zero matrix

        For each n in range (nmax):
            Stack price arrays P using p_c, p_2, p_1, c_h, c_p
            Extract w1 and w2 from W1 and W2
            Replace any negative values in w1, w2 with 0
            
            Simulate the optimal policy based on different cases:
                Case 1: OM dominates
                Case 2: CM dominates
                
                Calculate the reward R_star for each period based on the chosen policy
            
            Iterate over the periods in reverse order (Bellman equation):
                Update R_star based on the maximum reward from the previous period
            Store the minimum reward in Vo

        Find the maximum value in Vo and store in V_o

    Return the generated prices and Vo

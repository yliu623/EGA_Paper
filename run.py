import os
import itertools
import subprocess
import pandas as pd
from multiprocessing import Pool

depth_nn_hidden = [2,3,4,5,6,7]  
actor_lr = [0.0001, 0.00001, 0.000001]
critic_lr = [0.0001, 0.00001, 0.000001]
epsilon_decay = [0.994, 0.9994, 0.99994]
epsilon_min = [0.1, 0.01, 0.001, 0.0001]
batch_size = [64, 128, 256]
buffer_size = [60000, 600000, 6000000]
prioritized_replay_alpha = [0.6, 0.5]
prioritized_replay_beta0 = [0.4, 0.6, 0.8]
TAU = [0.0001, 0.00001]
num_sim = [2001, 4001]

all_args = [depth_nn_hidden, actor_lr, critic_lr, epsilon_decay, epsilon_min,  batch_size, buffer_size, prioritized_replay_alpha, prioritized_replay_beta0, TAU, num_sim]

combinations = list(itertools.product(*all_args))

num_rows = len(combinations)
num_col2_vals = 32

col1_vals = [(i // num_col2_vals) + 1 for i in range(num_rows)]
col2_vals = [(i % num_col2_vals) + 1 for i in range(num_rows)]

combinations = [(col1_vals[i], col2_vals[i], x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10]) for i, x in enumerate(combinations)]

df = pd.DataFrame(combinations, columns=["job", "worker", "depth_nn_hidden", "actor_lr", "critic_lr", "epsilon_decay", "epsilon_min", "batch_size", "buffer_size", "prioritized_replay_alpha","prioritized_replay_beta0","TAU", "num_sim"])
                                         
def run_single_process(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11):
    subprocess.run(['python', 'myddpg_per.py','-depth_nn_hidden', str(arg1), '-actor_lr',str(arg2), '-critic_lr', str(arg3), '-epsilon_decay', str(arg4), 
                    '-epsilon_min', str(arg5), '-batch_size', str(arg6),  '-buffer_size', str(arg7),'-prioritized_replay_alpha', str(arg8), '-prioritized_replay_beta0', str(arg9), '-TAU', str(arg10), '-num_sim', str(arg11)])

def process_row(x):
    run_single_process(*x)
    
def main(lsb_jobindex):
    df1 = df.loc[df['job'] == int(lsb_jobindex)]
    df2 = df1[["depth_nn_hidden", "actor_lr", "critic_lr", "epsilon_decay", "epsilon_min", "batch_size", "buffer_size", "prioritized_replay_alpha","prioritized_replay_beta0","TAU", "num_sim"]]
    with Pool(32) as p: # of worker
        p.map(process_row, df2.itertuples(index=False, name=None))

if __name__ == '__main__':
    main(os.environ["LSB_JOBINDEX"])
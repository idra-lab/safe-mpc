import numpy as np 
from safe_mpc.controller import SafeBackupController
from safe_mpc.set_verification_ocp import BackAndForthNStepControlInvariance_MAX
import time
import pickle
import numpy as np
from scipy.stats import qmc
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel
from safe_mpc.utils import get_ocp
from safe_mpc.robot_visualizer import RobotVisualizer
from tqdm import tqdm
from copy import deepcopy
import sys

args = parse_args()
start = args['start_array']
end = args['end_array']
model_name = args['system']
params = Parameters(args,model_name, rti=False)
model = AdamModel(params)
set_OCP = BackAndForthNStepControlInvariance_MAX(model)

# states_to_verify = np.load('data_results/sampled_states_large.npy')[start:end]
states_to_verify = np.load('data_results/N_step_unsafe_states.npy')[start:end]


max_horizon = 45
n_r_back_to_test = max_horizon
results = np.zeros((states_to_verify.shape[0],n_r_back_to_test,max_horizon), dtype=bool)
results[:] = False
results_N_step_CIS = np.load(f'data_results/verification_results_N_stepCIS.npy')

threshold_vel_success = 1e-2

progress_bar = tqdm(total=(end-start), desc=f'Testing initial conditions, alpha {params.alpha}')
for i in range(states_to_verify.shape[0]):

    for j in range(1,n_r_back_to_test):

        for k in range(j+1, max_horizon+1):  
            x_init = states_to_verify[i]

            result_problem, solution = set_OCP.solveProblem(x_init,j,k)  
            if result_problem:
                if (solution[model.nq:] - x_init[model.nq:]) <= 0 and (x_init[model.nq:] - solution[model.nq:]) <= threshold_vel_success:
                    results[i, j,k-1] = result_problem
                else:
                    results[i, j, k-1] = False
            else:
                results[i, j, k-1] = False
            print(f"State {i}_r_{j}_j_{k}_{x_init}: {'Safe' if results[i,j,k-1] else 'Unsafe'}")

    progress_bar.update(1)

np.save(f'data_results/verification_results_back_and_forth_within_N_CIS_ipopt_large_{start}_{end}.npy', results)
import numpy as np 
from safe_mpc.set_verification_ocp import BackAndForthNStepControlInvariance
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
set_OCP = BackAndForthNStepControlInvariance(model)

# states_to_verify = np.load('data_results/sampled_states_large.npy')[start:end]
states_to_verify = np.load('data_results/4dofN_step_unsafe_states.npy')[start:end]


max_horizon = 45
n_r_back_to_test = max_horizon
results = np.zeros((states_to_verify.shape[0],n_r_back_to_test,max_horizon), dtype=bool)
results[:] = False
# results_N_step_CIS = np.load(f'data_results/verification_results_N_stepCIS.npy')

progress_bar = tqdm(total=(end-start), desc=f'Testing initial conditions, alpha {params.alpha}')
for i in range(states_to_verify.shape[0]):
    # firstly compute all the results for j =45
    for j in range(1,n_r_back_to_test): 
        start_time = time.time()
        x_init = states_to_verify[i]
        result_problem, _ = set_OCP.solveProblem(x_init,j,max_horizon)  
        results[i,j,max_horizon-1] = result_problem
        end_time = time.time()
        print(f"State {i}_r_{j}_j_{max_horizon}_{x_init}: {'Safe' if results[i,j,max_horizon-1] else 'Unsafe'}")
        print(f"Time taken for state {i}_r_{j}_j_{max_horizon}: {end_time - start_time} seconds")

    # then for all the other horizons, but, if we find a safe one, we can skip the rest of the horizons for that state. If
    # a safe state was found for j_forward 45, we can skip all the horizons for j_forward < 45
    if not results[i,:,max_horizon-1].any(): # if no safe state was found for j_forward = 45, we can test the other horizons
        found_safe = False
    else:
        found_safe = True   

    for j in range(1,n_r_back_to_test):
        if found_safe:
            break
        for k in range(j+1, max_horizon):  
            x_init = states_to_verify[i]
            result_problem, _ = set_OCP.solveProblem(x_init,j,k)  
            results[i,j,k-1] = result_problem
            print(f"State {i}_r_{j}_j_{k}_{x_init}: {'Safe' if results[i,j,k-1] else 'Unsafe'}")
            if result_problem:
                    found_safe = True
                    break
    progress_bar.update(1)

np.save(f'data_results/4dofverification_results_back_and_forth_within_N_CIS_ipopt_large_start{start}_end{end}.npy', results)

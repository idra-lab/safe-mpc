import time
import pickle
import numpy as np
from tqdm import tqdm
from scipy.stats import qmc
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel
from safe_mpc.utils import  get_controller , get_ocp_acados, randomize_model
from safe_mpc.controller import  BackAndForthNstepController
from safe_mpc.robot_visualizer import RobotVisualizer
from safe_mpc.ocp import InverseKinematicsOCP
import copy
from safe_mpc.cost_definition import *

args = parse_args()
start = args['start_array']
end = args['end_array']
model_name = args['system']
params = Parameters(args,model_name, rti=False)
params.q_margin = args['joint_bounds_margin']
params.collision_margin = args['collision_margin']
params.build = args['build']
params.act = args['activation']
params.solver_type = 'SQP'
params.N=100
params.N = args['horizon']
params.noise_mass = args['noise']
params.noise_inertia = args['noise']
params.noise_cm = args['noise']

model = AdamModel(params)

params.cont_name = 'back_and_forth'

build_controllers = True

# ocp_with_net = BackAndForthNstepController(model, k_backward = 10)
# ocp_with_net.build_controller(build = build_controllers)

n_r_back_to_test = 44
max_horizon = 45
states_to_verify = np.load('data_results/sampled_states.npy')[start:end]

print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print(f'Shape of all states to verify: {states_to_verify.shape}')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

results = np.zeros((states_to_verify.shape[0],n_r_back_to_test,max_horizon), dtype=bool)
results[:] = np.nan

results_N_step_CIS = np.load(f'data_results/verification_results_N_CIS_acados.npy')
# to have minimum number of compilations, iterate on the horizons, and for each horizon iterate on the states
progress_bar = tqdm(total=((max_horizon-1)*max_horizon/2), desc=f'Testing initial conditions, alpha {params.alpha}')
for j in range(n_r_back_to_test):
    for k in range(max_horizon, j , -1):
        if j > 0:
            model = AdamModel(params)
            model.params.N = k
            ocp_with_net = BackAndForthNstepController(model, k_backward = j)
            ocp_with_net.build_controller(build = build_controllers)
        for i in range(states_to_verify.shape[0]):  
            x_k = states_to_verify[i]
            if j == 0: 
                results[i,j,:] = results_N_step_CIS[i,k-1]  
            else:
                status = ocp_with_net.solve(x_k)
                if (status == 0 or status == 2) and ocp_with_net.checkGuess():
                    results[i,j,k-1] = True
                else:
                    results[i,j,k-1] = False

            print(f"State {i}_r_{j}_j_{k}_{x_k}: {'Safe' if results[i,j,k-1] else 'Unsafe'}")
            progress_bar.update(1)
            
# for i in range(states_to_verify.shape[0]):
#     if (results[i,:] == False).all():
#             print(f'State {i}_{states_to_verify[i]}:  is Unsafe for all horizons')

np.save(f'data_results/verification_results_back_and_forth_within_N_CIS_acados_{start}_{end}.npy', results)
    

import time
import pickle
import numpy as np
from tqdm import tqdm
from scipy.stats import qmc
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel
from safe_mpc.utils import  get_controller , get_ocp_acados, randomize_model
from safe_mpc.robot_visualizer import RobotVisualizer
from safe_mpc.ocp import InverseKinematicsOCP
import copy
from safe_mpc.cost_definition import *
from safe_mpc.ocp import SafeAbortOCP
from safe_mpc.controller import SafeBackupController

args = parse_args()
model_name = args['system']
params = Parameters(args,model_name, rti=False)
params.q_margin = args['joint_bounds_margin']
params.collision_margin = args['collision_margin']
params.build = args['build']
params.act = args['activation']
params.solver_type = 'SQP'
params.N=100

params.noise_mass = args['noise']
params.noise_inertia = args['noise']
params.noise_cm = args['noise']

model = AdamModel(params)
cost = ZeroCost(model)

build_controllers = args['build']

ocp_ipopt = SafeAbortOCP(model)
ocp_ipopt.instantiateProblem()

ocp_acados = SafeBackupController(model)
ocp_acados.set_cost(cost)
ocp_acados.build_controller(build = build_controllers)
ocp_acados.resetHorizon(params.N)


states_to_verify = np.load('data_results/sampled_states.npy')

progress_bar = tqdm(total=states_to_verify.shape[0], desc=f'Testing if from initial conditions it is possible to stop the system')
start_time = time.time()
results_ipopt = np.zeros(states_to_verify.shape[0], dtype=bool)
results_acados = np.zeros(states_to_verify.shape[0], dtype=bool)

for j in range(states_to_verify.shape[0]):
    x_init = states_to_verify[j]
    # x_init[model.nq:] = np.zeros(model.nv) 
    # acados solution
    u0_g = np.array([np.zeros((model.nu,))]*params.N)
    x0_g = np.array([x_init]*(params.N +1))
    ocp_acados.setGuess(x0_g,u0_g)
    status = ocp_acados.solve(x_init)
    if (status == 0 or status == 2) and ocp_acados.checkGuess():
        results_acados[j] = True
    else:
        results_acados[j] = False

    # ipopt solution
    ocp_ipopt.setGuess(x0_g,u0_g)
    result, solution = ocp_ipopt.solve(x_init)
    results_ipopt[j] = result       


    print(f'State {x_init[:model.nq]} {j} is safe with acados? {results_acados[j]})')
    print(f'State {x_init[:model.nq]} {j} is safe with ipopt? {results_ipopt[j]}' )

    progress_bar.update(1)

np.save(f'data_results/backup_results_acados.npy', results_acados)
np.save(f'data_results/backup_results_ipopt.npy', results_ipopt)

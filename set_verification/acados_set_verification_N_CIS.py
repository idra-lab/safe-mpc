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
params.N=1

params.noise_mass = args['noise']
params.noise_inertia = args['noise']
params.noise_cm = args['noise']

model = AdamModel(params)

# cost = TrackingMovingCircleNLS(model,params.Q_weight,params.R_weight)
#cost = Tracking8NLS(model,params.Q_weight,params.R_weight)
cost = ZeroCost(model)
# cost = ReachTargetNLS(model,params.Q_weight,params.R_weight)

ocp_name = 'htwa'
params.cont_name = args['controller']

build_controllers = args['build']

ocp_with_net, controllers_list = get_ocp_acados(ocp_name, model)
ocp_with_net.set_cost(cost)
ocp_with_net.build_controller(build = build_controllers)
ocp_with_net.resetHorizon(params.N)


states_to_verify = np.load('data_results/sampled_states.npy')[start:end]

progress_bar = tqdm(total=states_to_verify.shape[0], desc=f'Testing initial conditions, alpha {ocp_with_net.model.params.alpha}')
start_time = time.time()
results = np.zeros(states_to_verify.shape[0], dtype=bool)
horizon_to_test = list((np.arange(1, 46)).astype(int))
horizon_to_test = list(range(1, 46))
results = np.zeros((states_to_verify.shape[0], len(horizon_to_test)),dtype=bool)


# rviz = RobotVisualizer(params, params.nq)
# if params.obs_flag:
#     rviz.addObstacles(params.obstacles)
# rviz.init_capsule(params.robot_capsules+params.obst_capsules)
# rviz.init_spheres(params.spheres_robot)


for i in range(states_to_verify.shape[0]):
    for (j,horizon) in enumerate(horizon_to_test):
        x_init = states_to_verify[i]
        ocp_with_net.resetHorizon(horizon)
        u0_g = np.array([np.zeros((model.nu,))]*horizon)
        x0_g = np.array([x_init]*(horizon+1))
        ocp_with_net.setGuess(x0_g,u0_g)
        status = ocp_with_net.solve(x_init)
        if (status == 0 or status == 2) and ocp_with_net.checkGuess():
            results[i, j] = True
        else:
            results[i, j] = False

        print(f'State {x_init[:model.nq]} {i} with horizon {horizon} is Safe? {results[i,j]}')
    progress_bar.update(1)
    if (results[i,:] == False).all():
        print(f'State {i}_{x_init}:  is Unsafe for all horizons')

np.save(f'data_results/verification_results_N_CIS_acados_{start}_{end}.npy', results)
print(f'succesful states = {np.sum(results.any(axis=1))} / {results.shape[0]}')
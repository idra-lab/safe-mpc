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


state_to_test = np.load('data_results/sampled_states_large.npy')

progress_bar = tqdm(total=state_to_test.shape[0], desc=f'Testing initial conditions, alpha {ocp_with_net.model.params.alpha}')
start_time = time.time()
results = np.zeros(state_to_test.shape[0], dtype=bool)

# rviz = RobotVisualizer(params, params.nq)
# if params.obs_flag:
#     rviz.addObstacles(params.obstacles)
# rviz.init_capsule(params.robot_capsules+params.obst_capsules)
# rviz.init_spheres(params.spheres_robot)


for i in range(state_to_test.shape[0]):
    # print(f'Safety value of state {i}: {ocp_with_net.safe_set.nn_func_x(state_to_test[i])}')
    x_init = state_to_test[i]
    ocp_with_net.resetHorizon(1)

    # rviz.displayWithEESphere(x_init[:params.nq],params.robot_capsules+params.obst_capsules,params.spheres_robot)
    # time.sleep(2)
    u0_g = np.array([np.zeros((model.nu,))]*ocp_with_net.N)
    x0_g = np.array([x_init]*(args['horizon']+1))
    ocp_with_net.setGuess(x0_g,u0_g)
    # x_init[ocp_with_net.model.nq:] = 0
    status = ocp_with_net.solve(x_init)
    if (status == 0 or status == 2) and ocp_with_net.checkGuess():
        results[i] = True
    else:        
        results[i] = False
    print(f'State {x_init[:params.nq]} {i} is Safe? {results[i]}')
    progress_bar.update(1)
np.save('data_results/verification_results_CIS_acados.npy', results)
print(f'succesful states = {np.sum(results)} / {results.shape[0]}')

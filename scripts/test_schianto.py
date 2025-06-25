import time
import pickle
import numpy as np
from tqdm import tqdm
from scipy.stats import qmc
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel
from safe_mpc.utils import  get_controller , get_ocp_acados
from safe_mpc.robot_visualizer import RobotVisualizer
from safe_mpc.ocp import InverseKinematicsOCP
import copy
from safe_mpc.cost_definition import *
from collections import defaultdict

args = parse_args()
model_name = args['system']
params = Parameters(model_name, rti=False)
params.build = args['build']
params.solver_type = 'SQP'
params.act = args['activation']
params.alpha = args['alpha']
model = AdamModel(params)

cost = TrackingMovingCircleNLS(model,params.Q_weight,params.R_weight)

ocp_name = 'naive'
params.cont_name = args['controller']

build_controllers = args['build']

ocp_with_net, controllers_list = get_ocp_acados(ocp_name, model)
ocp_with_net.set_cost(cost)
ocp_with_net.build_controller(False)

params.solver_type = 'SQP_RTI'
controller = get_controller(ocp_name, model)
cost_controller = TrackingMovingCircleNLS(model,params.Q_weight,params.R_weight)
cost_controller.set_solver_cost(controller)
controller.build_controller(False)

velocities = list(np.linspace(0.45,0.65,3))
horizons = [30,35,40,45]
fails=defaultdict(dict)

for vel in velocities:
    for hor in horizons:
        print(f'Test with horizon {hor} and velocity {vel}')

        ocp_with_net.resetHorizon(hor)


        InvKynSolver = InverseKinematicsOCP(model,ocp_with_net.cost.traj[:,0])
        solver_inv = InvKynSolver.instantiateProblem()
        sol = solver_inv.solve()
        x0 = sol.value(InvKynSolver.X[0])
        
        u0_g = np.array([np.zeros((model.nu,))]*ocp_with_net.N) 
        x0_g = np.array([x0]*(hor+1))
        ocp_with_net.setGuess(x0_g,u0_g)

        status = ocp_with_net.solve(x0)
        if (status == 0 or status==2) and ocp_with_net.checkGuess():
            print(f'Solver trajectory tracking initialization status {status}, x0 {x0}')
            xg_net = copy.copy(ocp_with_net.x_temp)
            ug_net = copy.copy(ocp_with_net.u_temp)
            print('SUCCESS')

        print('MPC')
        params.circle_center_vel = vel
        controller.resetHorizon(hor)
        controller.setGuess(xg_net, ug_net)
        controller.reset_controller()

        x_sim = x0
        for j in range(params.n_steps):
            u, sa_flag = controller.step(x_sim)
    
            tau = np.array([model.tau_fun(controller.x_temp[k], controller.u_temp[k]).T for k in range(len(controller.u_temp))])
            tau = np.array([model.tau_fun(controller.x_guess[k], controller.u_guess[k]).T for k in range(len(controller.u_guess))])
        
            x_sim, _ = model.integrate(x_sim, u)

            # Check next state bounds and collision
            if not model.checkStateBounds(x_sim):   
                print('  FAIL BOUNDS')
                print(f'\tState {j + 1} violation: {np.min(np.vstack((model.x_max - x_sim, x_sim - model.x_min)), axis=0)}')
                print(f'\tCurrent controller fails: {controller.fails}')
                fails[vel][hor] = True
                break
            if not controller.model.checkCollision(x_sim):
                print(f'  FAIL COLLISION at step {j + 1}')
                fails[vel][hor] = True
                break
            # Check convergence
            if j == params.n_steps -1:
                # if i in viable_idx:
                #     viable_idx.remove(i)
                fails[vel][hor] = False
                print('not converged')

        if np.linalg.norm(model.jointToEE(x_sim).T - model.ee_ref) < params.tol_conv:
            fails[vel][hor] = False
            print('  CONVERGED !!')

print(fails)
print(params.circle_offset_traj)

        
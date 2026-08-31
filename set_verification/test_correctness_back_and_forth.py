import numpy as np 
from safe_mpc.controller import SafeBackupController
from safe_mpc.set_verification_ocp import BackAndForthNStepControlInvariance, NStepControlInvarianceOCP
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
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
set_OCP_N_step = NStepControlInvarianceOCP(model)

states_to_verify = np.load('data_results/feasible_states_back_and_forth_2nd_run_2.npy')
r_j_feasible = np.load('data_results/indexes_to_test_posteriori_2nd_run.npy')

PLOT = False

for i in range(states_to_verify.shape[0]):
    # test state i, that from tests should be safe
    x0 = states_to_verify[i]

    # test for these horizons, that from analysis are the best ones
    r = r_j_feasible[i,0]
    j = r_j_feasible[i,1] +1

    # r=6
    # j=10

    print(f'\n\nTesting state {i} with initial condition {x0} at horizons r={r}, j={j}).') 
    
    result_problem, sol = set_OCP.solveProblem(x0, r, j)

    # try with N-step CIS at best horizon
    result_N_step_CIS, _ = set_OCP_N_step.solveProblem(x0, 24)

    # try with 1-step CIS
    result_1_step_CIS, _ = set_OCP_N_step.solveProblem(x0, 1)

    if not result_N_step_CIS:
        print(f'24_step failed')
    else:
        print(f'24_step succeeded')

    if not result_1_step_CIS:
        print(f'1_step failed')
    else:
        print(f'1_step succeeded')

    if not result_problem:
        print(f'Error: state {i} with initial condition {x0} is not safe for r={r}, j={j}! Try with r=10 j=11')
        result_problem, sol = set_OCP.solveProblem(x0, 10, 11)


    # test last state in safe set
    x_final = sol.value(set_OCP.X[-1])
    print(f'x0 solution {sol.value(set_OCP.X[0])}.')
    print(f'Initial state set evaluation {set_OCP.safe_set.nn_func_x(x0)} ')
    print(f'Last state in safe set? {set_OCP.safe_set.nn_func_x(x_final)} ')

    if set_OCP.safe_set.nn_func_x(x_final) < 0:
        print(f'Error: last state is not in safe set! Value: {set_OCP.safe_set.nn_func_x(x_final)}')

    # simulate trajectory with control solution to see if dynamics is respected
    traj = np.zeros((r+j+1, model.nx))
    tau = np.zeros((r+j, model.nu))
    traj[0] = x0
    for t_back in range(r):
        print(f'step: {t_back}')
        u_t = sol.value(set_OCP.U[t_back])
        tau[t_back] = np.array(model.tau_fun(traj[t_back], u_t)).squeeze()
        traj[t_back+1] = np.array(model.f_fun_back(traj[t_back], u_t)).squeeze()
    for t_forward in range(r, r+j):
        print(f'step: {t_forward}')
        u_t = sol.value(set_OCP.U[t_forward])
        tau[t_forward] = np.array(model.tau_fun(traj[t_forward], u_t)).squeeze()
        traj[t_forward+1] = np.array(model.f_fun(traj[t_forward], u_t)).squeeze()


    X_solver = np.array([sol.value(set_OCP.X[t]) for t in range(r+j+1)])


    if PLOT:
        # one subplot per state component
        time_idx = np.arange(r + j + 1)
        fig, axes = plt.subplots(model.nq, 1, figsize=(10, 2.5 * model.nx), sharex=True)

        if model.nx == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.plot(time_idx, traj[:,i], marker='o', linewidth=1.5, label ='simulated trajectory')
            ax.plot(time_idx, X_solver[:,i], 'g--', linewidth=1.5, label ='solver trajectory')
            ax.axvline(r, color='black', linestyle='--', linewidth=1.0, label='switch back->forward')
            ax.axhline(model.x_max[i], color='red', linestyle='--', linewidth=1.0, label='x_max')
            ax.axhline(model.x_min[i], color='red', linestyle='--', linewidth=1.0, label='x_min')
            ax.set_ylabel(f'x[{i}]')
            ax.grid(True, alpha=0.3)

        axes[0].legend(loc='best')
        axes[-1].set_xlabel('time step')
        fig.suptitle('Trajectory states (back-and-forth rollout)')
        fig.tight_layout()


        # one subplot per state component
        time_idx = np.hstack([np.arange(r, -1,-1), np.arange(1, j+1)])
        fig, axes = plt.subplots(model.nq, 1, figsize=(10, 2.5 * model.nx), sharex=True)

        if model.nx == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            traj_full = np.hstack([traj[:r+1, i][::-1], traj[r+1:, i]])
            X_solver_full = np.hstack([X_solver[:r+1, i][::-1], X_solver[r+1:, i]])
            
            ax.plot(time_idx, traj_full, marker='o', linewidth=1.5, label='simulated trajectory')
            ax.plot(time_idx, X_solver_full, 'g--', linewidth=1.5, label='solver trajectory')
            ax.axhline(model.x_max[i], color='red', linestyle='--', linewidth=1.0, label='x_max')
            ax.axhline(model.x_min[i], color='red', linestyle='--', linewidth=1.0, label='x_min')
            ax.axvline(r, color='black', linestyle='--', linewidth=1.0, label='switch back->forward')
            ax.set_ylabel(f'x[{i}]')
            ax.grid(True, alpha=0.3)

        axes[0].legend(loc='best')
        axes[-1].set_xlabel('time step')
        fig.suptitle('Trajectory states (back-and-forth rollout)')
        fig.tight_layout()

        # one subplot per torque component
        time_idx = np.arange(r + j)
        fig, axes = plt.subplots(model.nq, 1, figsize=(10, 2.5 * model.nx), sharex=True)

        if model.nx == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.plot(time_idx, tau[:,i], linewidth=1.5, label ='simulated trajectory')
            ax.axvline(r, color='black', linestyle='--', linewidth=1.0, label='switch back->forward')
            ax.axhline(model.tau_max[i], color='red', linestyle='--', linewidth=1.0, label='tau_max')
            ax.axhline(model.tau_min[i], color='red', linestyle='--', linewidth=1.0, label='tau_min')
            ax.set_ylabel(f'τ[{i}]')
            ax.grid(True, alpha=0.3)

        axes[0].legend(loc='best')
        axes[-1].set_xlabel('time step')
        fig.suptitle('Torque joints (back-and-forth rollout)')
        fig.tight_layout()

    x_0_forward= traj[r]
    forward_traj = [x_0_forward]
    for i in range(r-1, -1,-1):
        forward_traj.append(np.array(model.f_fun(forward_traj[-1], sol.value(set_OCP.U[i]))).squeeze())

    forward_traj = np.array(forward_traj)
    if PLOT:
        print(forward_traj)

    plt.show()




import time
import pickle
import numpy as np
from tqdm import tqdm
from scipy.stats import qmc
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel
from safe_mpc.utils import  get_controller , get_ocp_acados, RobotVisualizer
from safe_mpc.ocp import InverseKinematicsOCP
import copy

overwrite = True

args = parse_args()
model_name = args['system']
if model_name == 'z1':
    n_dofs = 4
elif model_name == 'fr3':
    n_dofs = 6
params = Parameters(model_name, rti=False)
params.build = args['build']
params.solver_type = 'SQP'
params.act = args['activation']
params.alpha = args['alpha']
model = AdamModel(params, n_dofs=n_dofs)

ocp_name = args['controller']
params.cont_name = args['controller']
ocp_with_net, controllers_list = get_ocp_acados(ocp_name, model)
ocp_with_net.resetHorizon(args['horizon'])
ocp_naive,_ = get_ocp_acados('naive',model)
ocp_naive.resetHorizon(args['horizon'])
ocp_zerovel,_ = get_ocp_acados('zerovel',model) 
ocp_zerovel.resetHorizon(args['horizon'])

num_ics = params.test_num
succ, fails, skip_ics = 0, 0, 0
sampler = qmc.Halton(model.nq, scramble=False)

x_guess_net, u_guess_net = [], []
x_guess_naive, u_guess_naive = [], []
x_guess_zerovel, u_guess_zerovel = [], []

rviz = RobotVisualizer(params, n_dofs=4)
if params.obs_flag:
    rviz.addObstacles(params.obstacles)
rviz.init_capsule(params.robot_capsules+params.obst_capsules)

progress_bar = tqdm(total=num_ics, desc=f'Generating initial conditions, alpha {ocp_with_net.model.params.alpha}')
start_time = time.time()
if not(ocp_with_net.model.params.track_traj):
    while succ < num_ics:
        
        q0 = qmc.scale(sampler.random(), model.x_min[:model.nq], model.x_max[:model.nq])[0]
        x0 = np.zeros((model.nx,))
        x0[:model.nq] = q0
    
        rviz.displayWithEESphere(x0[:n_dofs],params.robot_capsules+params.obst_capsules)
        if ocp_with_net.model.checkCollision(x0):
            print(f'accepted:{x0}')
            u0_g = np.array([np.zeros((model.nu,))]*ocp_with_net.N) 
            x0_g = np.array([x0]*(args['horizon']+1))
            ocp_with_net.setGuess(x0_g,u0_g)

            status = ocp_with_net.solve(x0)
            if (status == 0 or status==2) and ocp_with_net.checkGuess():
                print(f'Solver status {status}, x0 {x0}')
                xg_net = copy.copy(ocp_with_net.x_temp)
                ug_net = copy.copy(ocp_with_net.u_temp)
                x_guess_net.append(xg_net), u_guess_net.append(ug_net)
                succ += 1
                progress_bar.update(1)
                print('SUCCESS')
                if 'analytic' in args['controller']:
                    print(f'Check analytic constraint {ocp_with_net.model.checkAnalyticConstraints(ocp_with_net.x_temp[-1],ocp_with_net.obstacles)}')
                if args['controller'] == 'htwa':
                    print(f'Check network constraint {ocp_with_net.model.checkSafeConstraints(ocp_with_net.x_temp[-1])}')
                    print(f'Value: {ocp_with_net.model.nn_func(ocp_with_net.x_temp[-1], ocp_with_net.params.alpha)}')
                                
                ocp_naive.setGuess(x0_g,u0_g)
                ocp_zerovel.setGuess(x0_g,u0_g)
                status_naive = ocp_naive.solve(x0)
                if ((status_naive == 0 or status_naive == 2 ) and ocp_naive.checkGuess()):
                    x_guess_naive.append(copy.copy(ocp_naive.x_temp))
                    u_guess_naive.append(copy.copy(ocp_naive.u_temp))
                else: 
                    x_guess_naive.append(copy.copy(ocp_with_net.x_temp))
                    u_guess_naive.append(copy.copy(ocp_with_net.u_temp))

                status_zero_vel = ocp_zerovel.solve(x0)
                if ((status_zero_vel==0 or status_zero_vel==2) and ocp_zerovel.checkGuess()):
                    x_guess_zerovel.append(copy.copy(ocp_zerovel.x_temp))
                    u_guess_zerovel.append(copy.copy(ocp_zerovel.u_temp))
                else: 
                    x_guess_zerovel.append(copy.copy(ocp_with_net.x_temp))
                    u_guess_zerovel.append(copy.copy(ocp_with_net.u_temp))
                
            else:
                fails += 1
        else:
            print(f'Skipped:{x0}')
            skip_ics += 1
        time.sleep(3)
else:
    InvKynSolver = InverseKinematicsOCP(model,ocp_with_net.traj_to_track[:,0],obstacles)
    solver_inv = InvKynSolver.instantiateProblem()
    sol = solver_inv.solve()
    x0 = sol.value(InvKynSolver.X[0])
    # rviz.displayWithEESphere(x0[:ocp_with_net.model.nq],ocp_with_net.capsules)
    # rviz.display(x0[:ocp_with_net.model.nq])
    # rviz.addTraj(ocp_with_net.traj_to_track)
    u0_g = np.array([np.zeros((model.nu,))]*ocp_with_net.N) 
    x0_g = np.array([x0]*(args['horizon']+1))
    ocp_with_net.setGuess(x0_g,u0_g)

    status = ocp_with_net.solve(x0)
    if (status == 0 or status==2) and ocp_with_net.checkGuess():
        print(f'Solver trajectory tracking initialization status {status}, x0 {x0}')
        xg_net = copy.copy(ocp_with_net.x_temp)
        ug_net = copy.copy(ocp_with_net.u_temp)
        x_guess_net.append(xg_net), u_guess_net.append(ug_net)
        succ += 1
        print('SUCCESS')
        if 'analytic' in args['controller']:
            print(f'Check analytic constraint {ocp_with_net.model.checkAnalyticConstraints(ocp_with_net.x_temp[-1],ocp_with_net.obstacles)}')
        if args['controller'] == 'receding':
            print(f'Check network constraint {ocp_with_net.model.checkSafeConstraints(ocp_with_net.x_temp[-1])}')
            print(f'Value: {ocp_with_net.model.nn_func(ocp_with_net.x_temp[-1], ocp_with_net.params.alpha)}')
        
        ocp_naive.setGuess(x0_g,u0_g)
        ocp_zerovel.setGuess(x0_g,u0_g)
        status_naive = ocp_naive.solve(x0)
        if ((status_naive ==0 or status_naive == 2 ) and ocp_naive.checkGuess()):
            x_guess_naive.append(copy.copy(ocp_naive.x_temp))
            u_guess_naive.append(copy.copy(ocp_naive.u_temp))
        else: 
            x_guess_naive.append(copy.copy(ocp_with_net.x_temp))
            u_guess_naive.append(copy.copy(ocp_with_net.u_temp))

        status_zero_vel = ocp_zerovel.solve(x0)
        if ((status_zero_vel==0 or status_zero_vel==2) and ocp_zerovel.checkGuess()):
            x_guess_zerovel.append(copy.copy(ocp_zerovel.x_temp))
            u_guess_zerovel.append(copy.copy(ocp_zerovel.u_temp))
        else: 
            x_guess_zerovel.append(copy.copy(ocp_with_net.x_temp))
            u_guess_zerovel.append(copy.copy(ocp_with_net.u_temp))
    else:
        print('FAILED')

progress_bar.close()

print(f'Number of failed initializations: {fails}')
print(f'Number of skipped initial conditions: {skip_ics}')

traj__track = 'traj_track' if ocp_with_net.model.params.track_traj else "" 
with open(f'{params.DATA_DIR}{model_name}_naive_{args["horizon"]}hor_{int(params.alpha)}sm{traj__track}_guess.pkl', 'wb') as f:
        pickle.dump({'xg': np.asarray(x_guess_naive), 'ug': np.asarray(u_guess_naive)}, f)
with open(f'{params.DATA_DIR}{model_name}_zerovel_{args["horizon"]}hor_{int(params.alpha)}sm{traj__track}_guess.pkl', 'wb') as f:
        pickle.dump({'xg': np.asarray(x_guess_zerovel), 'ug': np.asarray(u_guess_zerovel)}, f)

if (args['controller']!= 'naive' and args['controller']!= 'zerovel'): 
    for cont in controllers_list:
        if cont in ['st','stwa','htwa','receding','parallel']:
            with open(f'{params.DATA_DIR}{model_name}_{cont}_{args["horizon"]}hor_{int(params.alpha)}sm{traj__track}_guess.pkl', 'wb') as f:
                pickle.dump({'xg': np.asarray(x_guess_net), 'ug': np.asarray(u_guess_net)}, f)


elapsed_time = time.time() - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)
print(f'Elapsed time: {hours}:{minutes:2d}:{seconds:2d}')

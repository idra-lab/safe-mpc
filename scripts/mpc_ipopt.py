import pickle
import numpy as np
from functools import reduce
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel
from safe_mpc.utils import get_ocp, randomize_model
from safe_mpc.controller import SafeBackupController
from safe_mpc.cost_definition import *
from safe_mpc.robot_visualizer import RobotVisualizer


test_num = 30

CALLBACK = True
VISUALIZE = False

n_iter_MPC = 150
ipopt_tol_MPC = 1e-5

n_iter_OCP = 500
ipopt_tol_OCP = 1e-6

args = parse_args()
model_name = args['system']
params = Parameters(args,model_name, rti=True)
params.q_margin = args['joint_bounds_margin']
params.collision_margin = args['collision_margin']

params.act = args['activation']
params.alpha = args['alpha']
params.backhor = args['back_hor']
horizon = args['horizon']
params.N=horizon
model = AdamModel(params)


model.ee_ref = params.ee_ref
nq = model.nq

print(f'\n q_min: {model.x_min},      q_max: {model.x_max} \n')

params.noise_mass = args['noise']
params.noise_inertia = args['noise']
params.noise_cm = args['noise']
params.control_noise = args['control_noise']

cont_name = args['controller']
controller = get_ocp(cont_name, model)

param_backup = Parameters(args,model_name, rti=True)
param_backup.q_margin = args['joint_bounds_margin']
param_backup.collision_margin = args['collision_margin']
param_backup.control_noise = args['control_noise']
param_backup.use_net = None
param_backup.N = args['back_hor']
param_backup.ipopt_opts['ipopt.max_iter'] = n_iter_MPC
param_backup.ipopt_opts['ipopt.tol'] = ipopt_tol_MPC
model_backup = AdamModel(param_backup)
safe_ocp = get_ocp('zerovel', model_backup)

traj__track = 'traj_track' if controller.model.params.track_traj else "" 

joint_margin=args["joint_bounds_margin"]
collision_margin=args["collision_margin"]
data = pickle.load(open(f'{params.DATA_DIR}{model_name}_{cont_name}_{horizon}hor_{int(params.alpha)}sm_use_net{controller.model.params.use_net}_{traj__track}_q_collision_margins_{controller.model.params.q_margin}_{controller.model.params.collision_margin}_guess.pkl', 'rb'))
print(f'{params.DATA_DIR}{model_name}_{cont_name}_{horizon}hor_{int(params.alpha)}sm_use_net{controller.model.params.use_net}_{traj__track}_q_collision_margins_{controller.model.params.q_margin}_{controller.model.params.collision_margin}_guess.pkl')

x_guess = data['xg']
u_guess = data['ug']
x_init = x_guess[:,0,:]


# MPC simulation 
conv_idx, collisions_idx, viable_idx = [], [], []
x_sim_list, u_list, x_viable = [], [], []
stats = []
traj_costs = [[] for _ in range(x_init.shape[0])]
EVAL = False

counters = np.zeros(5)
not_conv = 0
tau_viol = []
kp, kd = 0.1, 1e2
print(x_init.shape[0])

# Visualization
if VISUALIZE:
    rviz = RobotVisualizer(params, params.nq)
    if params.obs_flag:
        rviz.addObstacles(params.obstacles)
    rviz.init_capsule(params.robot_capsules+params.obst_capsules)
    rviz.init_spheres(params.spheres_robot)


for i in range(0,test_num):#x_init.shape[0]):
    model.reset_seed(i)
    model_backup.reset_seed(i)
    model.update_randomized_dynamics(controller_name=(f'noise{args["noise"]}_{i}'))
    model_backup.update_randomized_dynamics(controller_name=(f'noise{args["noise"]}_{i}'))
    #params.N = 500
    controller = get_ocp(cont_name, model)
    safe_ocp = get_ocp('zerovel', model_backup)
    params.ipopt_opts['ipopt.max_iter'] = n_iter_OCP
    params.ipopt_opts['ipopt.tol'] = ipopt_tol_OCP
    opti_controller= controller.instantiateProblem()
    opti_backup = safe_ocp.instantiateProblem()
    
    traj_costs[i] = 0
    print(f'Simulation {i + 1}/{test_num}')
    x0 = x_init[i]
    x_sim = np.empty((params.n_steps + 1, model.nx)) * np.nan
    u = np.empty((params.n_steps, model.nu)) * np.nan
    x_sim[0] = x0

    controller.reset_controller()

    #controller.setGuess(x_guess[i], u_guess[i])

    j = 0
    ja = 0
    sa_flag = False

    try:        
        controller.opti.set_value(controller.x_init, x0)
        sol = controller.opti.solve()
        controller.xg = np.array([sol.value(controller.X[k]) for k in range(controller.model.params.N + 1)])
        controller.ug = np.array([sol.value(controller.U[k]) for k in range(controller.model.params.N)])
        print(f'solved in {controller.opti.stats()["iter_count"]} iterations')

    except:
        controller.fails += 1
        print(f'Initialization_failed, use Acados guess')
        controller.setGuess(x_guess[i], u_guess[i])

    # Reinstantiate opti for MPC
    params.ipopt_opts['ipopt.max_iter'] = n_iter_MPC
    params.ipopt_opts['ipopt.tol'] = ipopt_tol_MPC
    opti_controller= controller.instantiateProblem()

    
    for j in range(params.n_steps):
        print(f'Step {j}', end=' ')
        # if controller.track_traj:
        #     traj_costs[i] += (controller.model.jointToEE(x_sim[-1])-controller.traj_to_track[:,1]).T @ controller.Q @ (controller.model.jointToEE(x_sim[-1])-controller.traj_to_track[:,1])
        if sa_flag and ja < safe_ocp.N:
            # Follow safe abort trajectory (PD to stabilize at the end)
            u[j] = u_abort[ja]
            ja += 1
            
        else:   
            u[j], sa_flag = controller.step(x_sim[j])
            # if args['controller'] in ['zerovel','receding','st','htwa']:
            #     if controller.ocp_solver.get_status() != 0:
            #         print(f'Status = {controller.ocp_solver.get_status()}')
            # print(f'Control at, problem {i} step {j} = {u[j]}')
    
            # Check Safe Abort
            if sa_flag:
                x_viable += [controller.getLastViableState()]
                if CALLBACK:
                    if params.urdf_name == 'z1':
                        print(f'  ABORT at step {j}, x = {x_viable[-1]}')
                        if controller.model.params.use_net:
                            print(f'  NN output at abort with current alpha {int(params.alpha)}: ' 
                                f'{controller.safe_set.nn_func_x(x_viable[-1])}')
                            print(f'  NN output at abort with alpha = 10: '
                                f'{controller.safe_set.nn_func(x_viable[-1], [0,0,0,10.,0])}')
                # Instead of breaking, solve safe abort problem
                xg = np.full((param_backup.N + 1, model.nx), x_viable[-1])
                ug = np.zeros((param_backup.N, model.nu))
                safe_ocp.setGuess(xg, ug)
                try:
                    status = safe_ocp.solve(x_viable[-1])
                    print(f' Number of SQP iterations: {controller.ocp_solver.get_stats("sqp_iter")}')
                except:
                    if CALLBACK:
                        print('  SAFE ABORT FAILED')
                        print('  Current controller fails:', controller.fails)
                    collisions_idx.append(i)
                    break
                ja = 0
                viable_idx.append(i)
                x_abort, u_abort = safe_ocp.x_temp, safe_ocp.u_temp
        
        x_sim[j + 1], _ = model.integrate(x_sim[j], u[j])
        
        if VISUALIZE:
            if j%20==0:
                rviz.displayWithEESphere(x_sim[j+1][:params.nq],params.robot_capsules+params.obst_capsules,params.spheres_robot)
        # print(f'EE position: {model.ee_fun(x_sim[j+1])}')


        # Check next state bounds and collision
        if not model.checkStateBounds(x_sim[j + 1]):   
            if CALLBACK:
                print('  FAIL BOUNDS')
                print(f'\tState {j + 1} violation: {np.min(np.vstack((model.x_max - x_sim[j + 1], x_sim[j + 1] - model.x_min)), axis=0)}')
                print(f'Solver failures {controller.fails}')
                print(f'State: {x_sim[j+1]}')

            collisions_idx.append(i)
            break
        if not controller.model.checkCollision(x_sim[j + 1]):
            collisions_idx.append(i)
            if CALLBACK:
                print(f'FAIL COLLISION at step {j + 1}')
                print(f'Solver failures {controller.fails}')
            break
        # Check convergence
        if j == params.n_steps -1:
            # if i in viable_idx:
            #     viable_idx.remove(i)
            not_conv+=1
            if CALLBACK:
                print('not converged')

    if np.linalg.norm(model.jointToEE(x_sim[-1]).T - model.ee_ref) < params.tol_conv:
        conv_idx.append(i)
        if CALLBACK:
            print('  SUCCESS !!')
        if i in viable_idx:
            viable_idx.remove(i)
    print(f'initial state: {x_sim[0]}')

    x_sim_list.append(x_sim), u_list.append(u)

viable_idx = [i for i in viable_idx if i not in collisions_idx]
viable_idx = list(set(viable_idx))
unconv_idx = np.setdiff1d(np.arange(test_num), 
                          reduce(np.union1d, (conv_idx, collisions_idx, viable_idx))).tolist()
print(f'Completed task: {len(conv_idx)}'
      f'\nCollisions: {len(collisions_idx)}'
      f'\nViable states: {len(viable_idx)}'
      f'\nNot converged: {test_num - len(conv_idx) - len(collisions_idx)}',
      f'\nNot failed: {not_conv}')

# Save simulation data
 
with open(f'{params.DATA_DIR}{model_name}_{cont_name}_use_net{controller.model.params.use_net}_{horizon}hor_{int(params.alpha)}sm_{traj__track}noise_{args["noise"]}_control_noise{args["control_noise"]}_q_collision_margins_{args["joint_bounds_margin"]}_{args["collision_margin"]}_IPOPT_mpc.pkl', 'wb') as f:
    pickle.dump({'x': np.asarray(x_sim_list),
                 'u': np.asarray(u_list),
                 'conv_idx' : conv_idx,
                 'collisions_idx' : collisions_idx,
                 'unconv_idx' : unconv_idx,
                 'viable_idx': viable_idx, 
                 'x_viable': np.asarray(x_viable)}, f)
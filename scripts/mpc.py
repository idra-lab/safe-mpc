import pickle
import numpy as np
from tqdm import tqdm
from functools import reduce
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel, TriplePendulumModel
from safe_mpc.utils import obstacles, ee_ref, get_controller


def convergenceCriteria(x, mask=None):
    if mask is None:
        mask = np.ones(model.nx)
    return np.linalg.norm(np.dot(mask, x - controller.x_ref)) < params.tol_conv


args = parse_args()
model_name = args['system']
params = Parameters(model_name, rti=True)
params.build = args['build']
params.act = args['activation']
if model_name == 'triple_pendulum':
    model = TriplePendulumModel(params)
else:
    model = AdamModel(params, n_dofs=4)
    model.ee_ref = ee_ref

cont_name = args['controller']
controller = get_controller(cont_name, model, obstacles)

data = pickle.load(open(f'{params.DATA_DIR}{model_name}_{cont_name}_guess.pkl', 'rb'))
x_guess = data['xg']
u_guess = data['ug']
x_init = x_guess[:,0,:]

# MPC simulation 
conv_idx, collisions_idx, viable_idx = [], [], []
x_sim_list, u_list, x_viable = [], [], []
stats = []
EVAL = False

counters = np.zeros(5)
tau_viol = []
for i in range(params.test_num):
    # controller.ocp_solver.reset()
    print(f'Simulation {i + 1}/{params.test_num}')
    x0 = x_init[i]
    x_sim = np.empty((params.n_steps + 1, model.nx)) * np.nan
    u = np.empty((params.n_steps, model.nu)) * np.nan
    nn_eval = np.empty(params.n_steps) * np.nan
    x_sim[0] = x0

    controller.setGuess(x_guess[i], u_guess[i])
    try:
        controller.r = controller.N
        controller.r_last = controller.N
    except:
        pass
    # if controller.ocp_name == 'receding':
    #     controller.r = controller.N
    #     controller.r_last = controller.N
    controller.fails = 0
    j = 0
    sa_flag = False
    for j in range(params.n_steps):
        u[j], sa_flag = controller.step(x_sim[j])
        # nn_eval[j] = model.nn_func(controller.x_temp[-1])

        tau = np.array([model.tau_fun(controller.x_temp[k], controller.u_temp[k]).T for k in range(len(controller.u_temp))])
        if not model.checkStateConstraints(controller.x_temp):
            counters[0] += 1
            if EVAL:
                print(f'\tx Bounds violated at step {j}')
                for k in range(len(controller.x_temp)):
                    viol = np.min(np.vstack((model.x_max - controller.x_temp[k], controller.x_temp[k] - model.x_min)), axis=0)
                    if np.any(viol + params.tol_x < 0):
                        print(f'\t\tState {k} out of bounds: {viol}')

        if not model.checkTorqueConstraints(tau):
            counters[1] += 1
            if EVAL:
                print(f'\ttau Bounds violated at step {j}')
                for k in range(len(tau)):
                    viol = model.tau_max - np.abs(tau[k])
                    if np.any(viol + params.tol_tau < 0):
                        # Collect any violation, taking the minimum (so max violation)
                        tau_viol.append(np.min(viol))
                        print(f'\t\tTorque {k} out of bounds: {viol}')
                
        if not np.all([controller.checkCollision(x) for x in controller.x_temp]):
            counters[2] += 1
            if EVAL:
                print(f'\tCollision at step {j}')
                for k in range(len(controller.x_temp)):
                    t_glob = model.jointToEE(controller.x_temp[k])
                    for obs in obstacles:
                        if obs['name'] == 'floor':
                            viol = t_glob[2] - obs['bounds'][0]
                            if viol + params.tol_obs < 0:
                                print(f'\t\tCollision {k} with floor: {viol}')
                        if obs['name'] == 'ball':
                            viol = np.sum((t_glob.flatten() - obs['position']) ** 2) - obs['bounds'][0]
                            if viol + params.tol_obs < 0:
                                print(f'\t\tCollision {k} with ball: {viol}')

        if cont_name not in ['naive', 'zerovel', 'trivial']:
            r = controller.r_last if cont_name == 'receding' else -1
            if not model.checkSafeConstraints(controller.x_temp[r]):
                counters[3] += 1
        if controller.last_status == 4:
            counters[4] += 1

        stats.append(controller.getTime())
        x_sim[j + 1], _ = model.integrate(x_sim[j], u[j])
        # Check Safe Abort
        if sa_flag:
            x_viable += [controller.getLastViableState()]
            viable_idx.append(i)
            print(f'  ABORT at step {j}, u = {u[j]}')
            break

        # Check next state bounds and collision
        if not model.checkStateConstraints(x_sim[j + 1]):   
            print('  FAIL BOUNDS')
            print(f'\tState {j + 1} violation: {np.min(np.vstack((model.x_max - x_sim[j + 1], x_sim[j + 1] - model.x_min)), axis=0)}')
            print(f'\tCurrent controller fails: {controller.fails}')
            collisions_idx.append(i)
            break
        if not controller.checkCollision(x_sim[j + 1]):
            collisions_idx.append(i)
            print('  FAIL COLLISION')
            break
    # Check convergence
    if not model.amodel.name == 'triple_pendulum':
        if np.linalg.norm(model.jointToEE(x_sim[-1]).T - ee_ref) < params.tol_conv:
            conv_idx.append(i)
            print('  SUCCESS !!')
    else:
        if convergenceCriteria(x_sim[-1], np.array([1, 0, 0, 1, 0, 0])) and not np.isnan(x_sim[-1]).any():
            conv_idx.append(i)
            print('  SUCCESS !!')
    x_sim_list.append(x_sim), u_list.append(u)

unconv_idx = np.setdiff1d(np.arange(params.test_num), 
                          reduce(np.union1d, (conv_idx, collisions_idx, viable_idx))).tolist()
print('Completed task: ', len(conv_idx))
print('Collisions: ', len(collisions_idx))
print('Viable states: ', len(viable_idx))
print('Not converged: ', len(unconv_idx))

print('Failing reasons:', 
      f'\n\t x bounds: {counters[0]}',
      f'\n\t tau bounds: {counters[1]}',
      f'\n\t Obstacle: {counters[2]}',
      f'\n\t Safe: {counters[3]}',
      f'\n\t Solver: {counters[4]}')

print('99% quantile of the computation time:')
times = np.array(stats)
for field, t in zip(controller.time_fields, np.quantile(times, 0.99, axis=0)):
    print(f"{field:<20} -> {t}")
    
# Save simulation data
with open(f'{params.DATA_DIR}{model_name}_{cont_name}_mpc.pkl', 'wb') as f:
    pickle.dump({'x': np.asarray(x_sim_list),
                 'u': np.asarray(u_list),
                 'conv_idx' : conv_idx,
                 'collisions_idx' : collisions_idx,
                 'unconv_idx' : unconv_idx,
                 'viable_idx': viable_idx, 
                 'x_viable': np.asarray(x_viable)}, f)

# print(f'Total torque violations: {len(tau_viol)}')
# np.save(f'{params.DATA_DIR}tau_viol_wo_check.npy', np.asarray(tau_viol))
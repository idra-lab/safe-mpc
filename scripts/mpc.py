import pickle
import numpy as np
from tqdm import tqdm
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
if model_name == 'triple_pendulum':
    model = TriplePendulumModel(params)
else:
    model = AdamModel(params, n_dofs=4)

cont_name = args['controller']
# if cont_name == 'naive':
#     model.setNNmodel()
controller = get_controller(cont_name, model, obstacles)
controller.setReference(ee_ref)

data = pickle.load(open(f'{params.DATA_DIR}{model_name}_{cont_name}_guess.pkl', 'rb'))
x_guess = data['xg']
u_guess = data['ug']
x_init = x_guess[:,0,:]

# MPC simulation 
conv_idx = []
collisions_idx = []
x_sim_list = []
# nn_list = []
stats = []
EVAL = False

counters = np.zeros(5)
# for i in tqdm(range(params.test_num), desc='MPC simulations'):
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
    if controller.ocp_name == 'receding':
        controller.r = controller.N
        controller.r_last = controller.N
    controller.fails = 0
    j = 0
    for j in range(params.n_steps):
        u[j] = controller.step(x_sim[j])
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

        if cont_name not in ['naive', 'zerovel']:
            r = controller.r_last if cont_name == 'receding' else -1
            if not model.checkSafeConstraints(controller.x_temp[r]):
                counters[3] += 1
        if controller.last_status == 4:
            counters[4] += 1

        stats.append(controller.getTime())
        x_sim[j + 1], _ = model.integrate(x_sim[j], u[j])
        # Check next state bounds and collision
        if not model.checkStateConstraints(x_sim[j + 1]):   
            collisions_idx.append(i)
            print('  FAIL BOUNDS')
            print(f'\tState {j + 1} violation: {np.min(np.vstack((model.x_max - x_sim[j + 1], x_sim[j + 1] - model.x_min)), axis=0)}')
            print(f'\tCurrent controller fails: {controller.fails}')
            if controller.ocp_name == 'receding':
                print(f'\tLast receding position before fail: {controller.r_last}')
            break
        if not controller.checkCollision(x_sim[j + 1]):
            # print(f'Obstacle detected at step {j + 1}, with state {x_sim[j + 1]}')
            collisions_idx.append(i)
            print('  FAIL COLLISION')
            break
        # Check convergence
        if not model.amodel.name == 'triple_pendulum':
            if np.linalg.norm(model.jointToEE(x_sim[j + 1]).T - ee_ref) < params.tol_conv:
                conv_idx.append(i)
                print('  SUCCESS !!')
                break
        else:
            if convergenceCriteria(x_sim[j + 1], np.array([1, 0, 0, 1, 0, 0])):
                conv_idx.append(i)
                print('  SUCCESS !!')
                break
    x_sim_list.append(x_sim)#, nn_list.append(nn_eval)

unconv_idx = np.setdiff1d(np.arange(params.test_num), np.union1d(conv_idx, collisions_idx))
print('Completed task: ', len(conv_idx))
print('Collisions: ', len(collisions_idx))
print('Not converged: ', len(unconv_idx))
# print('Convergence indices: ', conv_idx)
# print('Collision indices: ', collisions_idx)

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
with open(f'{params.DATA_DIR}{cont_name}_mpc.pkl', 'wb') as f:
    pickle.dump({'x': np.asarray(x_sim_list),
                #  'nn_eval': np.asarray(nn_list),
                 'conv_idx' : np.asarray(conv_idx),
                 'collisions_idx' : np.asarray(collisions_idx),
                 'unconv_idx' : np.asarray(unconv_idx)}, f)

# print(f'Total torque violations: {len(tau_viol)}')
# np.save(f'{params.DATA_DIR}tau_viol_wo_check.npy', np.asarray(tau_viol))
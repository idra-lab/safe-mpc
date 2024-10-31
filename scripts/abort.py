import pickle
import numpy as np
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel
from safe_mpc.utils import obstacles
from safe_mpc.controller import SafeBackupController


args = parse_args()
model_name = args['system']
params = Parameters(model_name, rti=False)
params.build = args['build']
params.act = args['activation']
params.N = args['horizon']
params.alpha = args['alpha']
model = AdamModel(params, n_dofs=4)
model.setNNmodel()
cont_name = args['controller']
controller = SafeBackupController(model, obstacles)

# file_prefix = f'{params.DATA_DIR}{model_name}_{cont_name}'
file_prefix = f'{params.DATA_DIR}mh/{model_name}_{cont_name}_{params.N}hor'
# file_prefix = f'{params.DATA_DIR}/ma/{model_name}_{cont_name}_{int(params.alpha)}'
data = pickle.load(open(f'{file_prefix}_mpc.pkl', 'rb'))
x = data['x']
u = data['u']
x_viable, viable_idx = data['x_viable'], data['viable_idx']
collisions_idx, unconv_idx = data['collisions_idx'], data['unconv_idx']
n = len(x_viable)
H_b = np.eye(4)

if cont_name in ['stwa', 'htwa', 'receding']:
    stats = []
    succ = 0
    for i, idx in enumerate(viable_idx):
        print(f'Simulation {i + 1}/{n}')
        print(f'\tNN output: {model.nn_func(x_viable[i])}')

        x_guess = np.full((params.N + 1, model.nx), x_viable[i])
        u_guess = np.zeros((params.N, model.nu))
        controller.setGuess(x_guess, u_guess)

        status = controller.solve(x_viable[i])
        if status == 0:
            succ += 1
            x_abort , u_abort = controller.x_temp, controller.u_temp
            stats.append(controller.ocp_solver.get_stats('time_tot'))
            print(f'\tSuccessful solution in {stats[-1]:.4f} seconds')
            unconv_idx.append(idx)
            # Find first nan in x (where we apply abort trajectory)
            j = np.where(np.isnan(x[idx][:,0]))[0][0] - 1
            if j + params.N <= params.n_steps:
                x[idx, j:j + params.N + 1] = x_abort
                u[idx, j:j + params.N] = u_abort
                # Repeat last state and control for the rest of the trajectory
                x[idx, j + params.N + 1:] = x_abort[-1]
                u[idx, j + params.N:] = u_abort[-1]
            else:
                j_term = params.n_steps - j
                x[idx, j:] = x_abort[:j_term + 1]
                u[idx, j:] = u_abort[:j_term]
            if model.checkRunningConstraints(x[idx], u[idx]) and \
               np.all(controller.checkCollision(x_single) for x_single in x[idx]):
                print('\tSolution is feasible')
                # Check dynamics feasibility
                if model.checkDynamicsConstraints(x[idx], u[idx]):
                    print('\t\tDynamics feasibility check passed')
                else:
                    print('\t\tFailed dynamics feasibility check')
            else:
                print('\tFailed feasibility check, something went wrong')
        else:
            print('\tFailed solution, status:', status)
            collisions_idx.append(idx)

    print('\nCollisions: ', len(collisions_idx))
    print('Unconverged: ', len(unconv_idx))
    if len(stats) > 0: 
        stats = np.array(stats)
        print(f'Success rate: {succ}/{n}')
        t_mean = np.mean(stats)
        t_quant = np.quantile(stats, 0.99)
        print(f'Mean time: {t_mean:.4f}\n99% quantile time: {t_quant:.4f}')

with open(f'{file_prefix}_res.pkl', 'wb') as f:
    pickle.dump(data, f)

    
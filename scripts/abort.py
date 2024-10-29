import time
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
model = AdamModel(params, n_dofs=4)
model.setNNmodel()
cont_name = args['controller']
controller = SafeBackupController(model, obstacles)

data = pickle.load(open(f'{params.DATA_DIR}{model_name}_{cont_name}_mpc.pkl', 'rb'))
x = data['x']
u = data['u']
x_viable, viable_idx = data['x_viable'], data['viable_idx']
collisions_idx, unconv_idx = data['collisions_idx'], data['unconv_idx'].tolist()
n = len(x_viable)
H_b = np.eye(4)

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
        j = np.where(np.isnan(x[idx][:,0]))[0][0]
        x[idx, j:j + params.N + 1] = x_abort
        u[idx, j:j + params.N] = u_abort
        # Repeat last state and control for the rest of the trajectory
        x[idx, j + params.N + 1:] = x_abort[-1]
        u[idx, j + params.N:] = u_abort[-1]
        for k in range(params.n_steps + 1):
            print(f't: {k * params.dt:.2f}\tx: {x[idx, k]}')
    else:
        print('\tFailed solution, status:', status)
        collisions_idx.append(viable_idx[i])

print('\nCollisions: ', len(collisions_idx))
print('Unconverged: ', len(unconv_idx))

stats = np.array(stats)
print(f'Success rate: {succ}/{n}')
t_mean = np.mean(stats)
t_quant = np.quantile(stats, 0.99)
print(f'Mean time: {t_mean:.4f}\n99% quantile time: {t_quant:.4f}')

with open(f'{params.DATA_DIR}{model_name}_{cont_name}_mpc.pkl', 'wb') as f:
    pickle.dump(data, f)

    
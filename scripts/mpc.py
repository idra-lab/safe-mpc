import pickle
import numpy as np
from tqdm import tqdm
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel
from safe_mpc.utils import obstacles, ee_ref, get_controller


args = parse_args()
params = Parameters('z1', rti=True)
params.build = args['build']
model = AdamModel(params, n_dofs=4)

cont_name = args['controller']
controller = get_controller(cont_name, model, obstacles)
controller.setReference(ee_ref)

data = pickle.load(open(f'{params.DATA_DIR}{cont_name}_guess.pkl', 'rb'))
x_guess = data['xg']
u_guess = data['ug']
x_init = x_guess[:,0,:]

# MPC simulation 
conv_idx = []
collisions_idx = []
x_sim_list = []
stats = []

for i in tqdm(range(params.test_num), desc='MPC simulations'):
# for i in range(params.test_num):
    # print(f'Simulation {i + 1}/{params.test_num}')
    x0 = x_init[i]
    x_sim = np.empty((params.n_steps + 1, model.nx)) * np.nan
    u = np.empty((params.n_steps, model.nu)) * np.nan
    x_sim[0] = x0

    controller.setGuess(x_guess[i], u_guess[i])
    controller.fails = 0
    j = 0
    for j in range(params.n_steps):
        u[j] = controller.step(x_sim[j])
        stats.append(controller.getTime())
        x_sim[j + 1] = model.integrate(x_sim[j], u[j])
        # Check next state bounds and collision
        if not model.checkStateConstraints(x_sim[j + 1]):
            collisions_idx.append(i)
            break
        if not controller.checkCollision(x_sim[j + 1]):
            collisions_idx.append(i)
            break
    # Check convergence
    if np.linalg.norm(model.jointToEE(x_sim[-1]).T - ee_ref) < params.conv_tol:
        conv_idx.append(i)
    x_sim_list.append(x_sim)

unconv_idx = np.setdiff1d(np.arange(params.test_num), np.union1d(conv_idx, collisions_idx))
print('Completed task: ', len(conv_idx))
print('Collisions: ', len(collisions_idx))
print('Not converged: ', len(unconv_idx))
# print('Convergence indices: ', conv_idx)
# print('Collision indices: ', collisions_idx)

print('99% quantile of the computation time:')
times = np.array(stats)
for field, t in zip(controller.time_fields, np.quantile(times, 0.99, axis=0)):
    print(f"{field:<20} -> {t}")

# Save simulation data
with open(f'{params.DATA_DIR}{cont_name}_mpc.pkl', 'wb') as f:
    pickle.dump({'x': np.asarray(x_sim_list),
                 'conv_idx' : np.asarray(conv_idx),
                 'collisions_idx' : np.asarray(collisions_idx),
                 'unconv_idx' : np.asarray(unconv_idx)}, f)
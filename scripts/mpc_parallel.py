import time
import pickle
import numpy as np
import multiprocessing as mp
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel
from safe_mpc.utils import obstacles, ee_ref, get_controller


def simulate_mpc(xg, ug):
    x_sim = np.full((params.n_steps + 1, model.nx), np.nan)
    u_sim = np.empty((params.n_steps, model.nu))
    x_sim[0] = xg[0]

    controller.setGuess(xg, ug)
    try:
        controller.r = controller.N
        controller.r_last = controller.N
    except:
        pass
    controller.fails = 0
    stats = np.zeros(3, dtype=int)         # [conv, viable, collisions]
    j = 0
    for j in range(params.n_steps):
        u_sim[j] = controller.step(x_sim[j])
        x_sim[j + 1], _ = model.integrate(x_sim[j], u_sim[j])
        # Check next state bounds and collision
        if not model.checkStateConstraints(x_sim[j + 1]):
            if np.isnan(x_sim[j + 1]).any():
                # x_viable += [controller.getLastViableState()]
                stats[1] += 1
            else:
                stats[2] += 1
            break
        if not controller.checkCollision(x_sim[j + 1]):
            stats[2] += 1
            break
    if np.linalg.norm(model.jointToEE(x_sim[-1]).T - ee_ref) < params.tol_conv \
        and not np.isnan(x_sim[-1]).any():
        stats[0] += 1
    return x_sim, u_sim, stats


args = parse_args()
model_name = args['system']
params = Parameters(model_name, rti=True)
params.build = args['build']
params.act = args['activation']
model = AdamModel(params, n_dofs=4)

cont_name = args['controller']
controller = get_controller(cont_name, model, obstacles)
controller.setReference(ee_ref)

data = pickle.load(open(f'{params.DATA_DIR}{model_name}_{cont_name}_guess.pkl', 'rb'))
x_guess = data['xg']
u_guess = data['ug']

start_time = time.time()
controller.solve(np.zeros(model.nx))
args_mpc = [(x_guess[i], u_guess[i]) for i in range(params.test_num)]
with mp.Pool(params.cpu_num) as pool:
    results = pool.starmap(simulate_mpc, args_mpc)

# results = []
# for i in range(params.test_num):
#     results.append(simulate_mpc(x_guess[i], u_guess[i]))

x_list, u_list, stats_list = zip(*results)
tot_stats = np.sum(np.array(stats_list), axis=0)    # [conv, viable, collisions]
unconv = params.test_num - np.sum(tot_stats)
print(f'Converged: {tot_stats[0]}, Viable: {tot_stats[1]}, Collisions: {tot_stats[2]}, Unconverged: {unconv}')

# print('99% quantile of the computation time:')
# times = np.array(stats)
# for field, t in zip(controller.time_fields, np.quantile(times, 0.99, axis=0)):
#     print(f"{field:<20} -> {t}")

# # Save simulation data
# with open(f'{params.DATA_DIR}{model_name}_{cont_name}_mpc.pkl', 'wb') as f:
#     pickle.dump({'x': np.asarray(x_sim_list),
#                  'u': np.asarray(u_list),
#                  'conv_idx' : np.asarray(conv_idx),
#                  'collisions_idx' : np.asarray(collisions_idx),
#                  'unconv_idx' : np.asarray(unconv_idx),
#                  'viable_idx': np.asarray(viable_idx),
#                  'x_viable': np.asarray(x_viable)}, f)

elapsed_time = time.time() - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)
print(f'Elapsed time: {hours}:{minutes:2d}:{seconds:2d}')
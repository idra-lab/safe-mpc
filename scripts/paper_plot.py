import pickle
import numpy as np
import matplotlib.pyplot as plt
from safe_mpc.parser import Parameters
from safe_mpc.model import TriplePendulumModel
from safe_mpc.abstract import SimDynamics
from safe_mpc.controller import STController, STWAController, SafeBackupController
from safe_mpc.plot_utils import PlotUtils


def evaluate_nn(x):
    outputs = np.empty(x.shape[0])
    for p in range(x.shape[0]):
        outputs[p] = model.nn_func(x[p], conf.alpha)
    return outputs


def convergenceCriteria(x, mask=None):
    if mask is None:
        mask = np.ones(model.nx)
    return np.linalg.norm(np.dot(mask, x - model.x_ref)) < conf.conv_tol


conf = Parameters('triple_pendulum', 'stwa')
conf.alpha = 10
model = TriplePendulumModel(conf)
simulator = SimDynamics(model)
naive_MPC = STController(simulator)
controller = STWAController(simulator)
naive_MPC.setReference(model.x_ref)
controller.setReference(model.x_ref)
# Change the horizon for the abort task
conf = Parameters('triple_pendulum', 'abort', rti=False)
conf.alpha = 10
backup = SafeBackupController(SimDynamics(TriplePendulumModel(conf)))

# Load data
data = pickle.load(open(conf.ROOT_DIR + '/data/st_results.pkl', 'rb'))
data_wa = pickle.load(open(conf.ROOT_DIR + '/data/stwa_results.pkl', 'rb'))

x_naive = data['x_sim']
idx = data_wa['idx_abort']
print('Index of the tests that cannot be solved (NAIVE): ', data['idx_abort'])
print('Index of the tests that cannot be solved (STWA): ', idx)
print('Max number of iter i: ', len(idx))

# Select the test to plot
i = 22            
# Nice ones: 1, 2, 4, 5, 6, 7, 15, 17, 19, 21, 22
# Cannot be solved: 9, 10, 11, 18, 24       
# Problem: 0, 3, 8, 13, 15, 16, 20, 25
# Big PROBLEM: 12, 23 
h = idx[i]
print(f'Iter number: {i}, Test number: {h}')
x0_vec = np.load(conf.DATA_DIR + 'x_init_10.npy')
x_guess_st = np.load(conf.DATA_DIR + 'st_x_guess.npy')
u_guess_st = np.load(conf.DATA_DIR + 'st_u_guess.npy')
x_guess_vec = np.load(conf.DATA_DIR + 'stwa_x_guess.npy')
u_guess_vec = np.load(conf.DATA_DIR + 'stwa_u_guess.npy')

k = 0
u = np.empty((conf.n_steps, model.nu)) * np.nan
x_sim = np.empty((conf.n_steps + 1, model.nx)) * np.nan
x_sim[0] = np.hstack([x0_vec[h], np.zeros(model.nv)])
x_N = np.empty_like(x_sim) * np.nan
x_naive = np.copy(x_sim)
u_naive = np.copy(u)

print('Simulating naive MPC...')
kk = 0
naive_MPC.setGuess(x_guess_st[h], u_guess_st[h])
for kk in range(conf.n_steps):
    u_naive[kk] = naive_MPC.step(x_naive[kk])
    x_naive[kk + 1] = simulator.simulate(x_naive[kk], u_naive[kk])             #naive_MPC.x_guess[0]
    if not model.checkStateConstraints(x_naive[kk + 1]) or np.isnan(u_naive[kk]).any():
        print(u_naive[kk])
        print(x_naive[kk + 1])
        print(model.checkStateConstraints(x_naive[kk + 1]))
        break
    if convergenceCriteria(x_naive[kk + 1], np.array([1, 0, 0, 1, 0, 0])):
        break

print('Simulating MPC with abort...')
controller.setGuess(x_guess_vec[h], u_guess_vec[h])
for k in range(conf.n_steps):
    u[k] = controller.step(x_sim[k])
    x_N[k] = np.copy(controller.x_temp[-1])
    x_sim[k + 1] = controller.x_guess[0]
    if not model.checkStateConstraints(x_sim[k + 1]) or np.isnan(u[k]).any():
        break
    if convergenceCriteria(x_sim[k + 1], np.array([1, 0, 0, 1, 0, 0])):
        break

# Find the abort trajectory with the backup controller
print('Simulating backup controller...')
x_viable = controller.getLastViableState()
backup.setGuess(np.full((backup.N + 1, model.nx), x_viable),
                np.zeros((backup.N, model.nu)))
status = backup.solve(x_viable)
x_abort = np.empty((backup.N + 1, model.nx)) * np.nan
if status == 0:
    x_abort = backup.x_temp
    u_abort = backup.u_temp
    print('Abort trajectory found!')
    print('Last state: ', x_abort[-1])

# Plot the results
util = PlotUtils(conf)
nn_out = evaluate_nn(x_N[:k])
util.plot_task_abortion(k, controller.N, x_naive[~np.isnan(x_naive[:, 0])], x_sim, x_abort, nn_out)
plt.savefig(util.plot_dir + f'/task_abortion_{i}.pdf')

# plt.show()


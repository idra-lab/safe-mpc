import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from safe_mpc.parser import Parameters
from safe_mpc.model import TriplePendulumModel
from safe_mpc.abstract import SimDynamics
from safe_mpc.controller import RecedingController
from debug import Debug


def evaluate_nn(x):
    n = np.shape(x)[0]
    out_vec = np.zeros((n,))
    for h in range(n):
        out_vec[h] = model.nn_func(x[h], conf.alpha)
    return out_vec


conf = Parameters('../config/params.yaml')
model = TriplePendulumModel(conf)
simulator = SimDynamics(model)
ocp = RecedingController(simulator)
debugger = Debug(conf, ocp)

sampler = qmc.Halton(d=ocp.ocp.dims.nu, scramble=False)
sample = sampler.random(n=conf.test_num)
eps = 1e-5
l_bounds = model.x_min[:model.nq] + eps
u_bounds = model.x_max[:model.nq] - eps
x0_vec = qmc.scale(sample, l_bounds, u_bounds)

x_ref = np.array([conf.q_max - 0.05, np.pi, np.pi, 0, 0, 0])
ocp.setReference(x_ref)

for i in range(conf.N):
    ocp.ocp_solver.cost_set(i, "zl", np.zeros((1,)) * i / conf.N * conf.ws_t)
# ocp.ocp_solver.cost_set(conf.N, "zl", np.ones((1,)) * conf.ws_t)  HARD TERMINAL


def init_guess(p):
    x0 = np.zeros((model.nx,))
    x0[:model.nq] = x0_vec[p]

    status = ocp.initialize(x0)
    print(status)
    if status == 0:
        print(ocp.ocp_solver.get_residuals())
    return ocp.getGuess()


success = 0
x_guess_vec, u_guess_vec = [], []
for i in range(conf.test_num):
    xg, ug = init_guess(i)
    nn_out = evaluate_nn(xg)
    if nn_out[-1] > 0.:
        success += 1
    print('Test #', i, 'nn_out: ', nn_out[-1])
    x_guess_vec.append(xg)
    u_guess_vec.append(ug)

print('Init guess success: ' + str(success) + ' over ' + str(conf.test_num))

# PLOT
t = np.arange(0, (conf.N + 1) * conf.dt, conf.dt)
for i in range(conf.test_num):
    fig, ax = plt.subplots(4, 1, sharex='col')
    for j in range(3):
        ax[j].plot(t, x_guess_vec[i][:, j], label='q' + str(j + 1), color='darkblue', linewidth=1.5)
        debugger.plotBounds(t, ax[j])
    ax[3].plot(t, evaluate_nn(x_guess_vec[i]), label='nn', color='darkorange', linewidth=1.5)
    ax[0].set_ylabel('q1 (rad)')
    ax[1].set_ylabel('q2 (rad)')
    ax[2].set_ylabel('q3 (rad)')
    ax[3].set_ylabel('NN out (rad/s)')
    ax[3].set_xlabel('Time (s)')
    plt.savefig('plots/test' + str(i) + '.png')
    plt.close()

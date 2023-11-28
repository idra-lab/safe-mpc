import numpy as np
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
from scipy.stats import qmc
from safe_mpc.parser import Parameters
from safe_mpc.model import TriplePendulumModel
from safe_mpc.abstract import SimDynamics
from safe_mpc.controller import STController
from debug import Debug


def evaluate_nn(x):
    n = np.shape(x)[0]
    out_vec = np.zeros((n,))
    for hh in range(n):
        out_vec[hh] = model.nn_func(x[hh], conf.alpha)
    return out_vec


ENABLE_PRINTING = False
conf = Parameters('../config/params.yaml')
model = TriplePendulumModel(conf)
simulator = SimDynamics(model)
ocp = STController(simulator)
# ocp = HTWAController(simulator)
ocp.ocp_solver.set(conf.N, "p", conf.alpha)
debugger = Debug(conf, ocp)

sampler = qmc.Halton(d=ocp.ocp.dims.nu, scramble=False)
sample = sampler.random(n=conf.test_num)
eps = 1e-5
l_bounds = model.x_min[:model.nq] + eps
u_bounds = model.x_max[:model.nq] - eps
x0_vec = qmc.scale(sample, l_bounds, u_bounds)

x_ref = np.array([conf.q_max - 0.05, np.pi, np.pi, 0, 0, 0])
ocp.setReference(x_ref)

num_iter = 18
sat_values = np.linspace(0.2, 1, num_iter)
t = np.arange(0, (conf.N + 1) * conf.dt, conf.dt)
success = 0
save_ICs = []
for h in range(conf.test_num):
    print('Test #', h)
    # Reset the solver and the ICs for each test
    ocp.ocp_solver.reset()
    ws_t = 1e-10
    x_guess_vec, u_guess_vec = [], []
    x0 = np.zeros((model.nx,))
    x0[:model.nq] = x0_vec[h]
    ocp.setGuess(np.full((ocp.N + 1, model.nx), x0), np.zeros((ocp.N, model.nu)))
    for k in range(num_iter):
        ocp.ocp_solver.cost_set(ocp.N, "zl", ws_t * np.ones((1,)))
        status = ocp.solve(x0)
        res = ocp.ocp_solver.get_residuals()
        if ENABLE_PRINTING:
            print('----> Iteration #', k, 'ws_t: ', ws_t, ' <----')
            print('Status: ', status)
            print('Bound constraint: ', model.checkRunningConstraints(ocp.x_temp, ocp.u_temp))
            print('Dynamics constraint: ', simulator.checkDynamicsConstraints(ocp.x_temp, ocp.u_temp))
            print('NN output: ', model.nn_func(ocp.x_temp[-1], conf.alpha))
            print('Constraint violation: ', ocp.ocp_solver.get(ocp.N, "sl")[0])
            print('Residuals:\nStat: %5.f\tEq: %.5f\tIneq: %.5f\tComp: %.5f' % (res[0], res[1], res[2], res[3]))
            print('-----------------------------------------------')
        if status != 4:
            x_guess_vec.append(np.copy(ocp.x_temp))
            u_guess_vec.append(np.copy(ocp.u_temp))
            ocp.setGuess(ocp.x_temp, ocp.u_temp)
        ws_t *= 10

    fig, ax = plt.subplots(4, 1, sharex='col')
    nn_out = None
    for i in range(len(x_guess_vec)):
        blues = hsv_to_rgb((0.6, sat_values[i], 1))
        reds = hsv_to_rgb((0, sat_values[i], 1))
        for j in range(3):
            ax[j].plot(t, x_guess_vec[i][:, j], color=blues)
            debugger.plotBounds(t, ax[j])
        nn_out = np.reshape(evaluate_nn(x_guess_vec[i]), (ocp.N + 1,))
        ax[3].plot(t, nn_out, color=reds)
    ax[0].set_ylabel('q1 (rad)')
    ax[1].set_ylabel('q2 (rad)')
    ax[2].set_ylabel('q3 (rad)')
    ax[3].set_ylabel('NN out (rad/s)')
    ax[3].set_xlabel('Time (s)')
    plt.savefig('plots/test' + str(h) + '.png')
    plt.close()
    if nn_out[-1] > 0.:
        success += 1
        save_ICs.append(h)

print('Init guess success: ' + str(success) + ' over ' + str(conf.test_num))
# np.save('ics_iterative_soft.npy', np.array([save_ICs]))

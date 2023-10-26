import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import pendulum_conf as conf
from safe_mpc.model import TriplePendulumModel
from safe_mpc.abstract import SimDynamics
from safe_mpc.controller import NaiveController

model = TriplePendulumModel(conf)
simulator = SimDynamics(conf, model)
ocp = NaiveController(conf, model)

test_num = 100
sampler = qmc.Halton(d=ocp.ocp.dims.nu, scramble=False)
sample = sampler.random(n=test_num)
l_bounds = model.x_min[:model.nq]
u_bounds = model.x_max[:model.nq]
x0_vec = qmc.scale(sample, l_bounds, u_bounds)

t = np.arange(0, conf.T, conf.dt)
x_ref = np.array([conf.q_max - 0.05, np.pi, np.pi, 0, 0, 0])
ocp.setReference(x_ref)

def init_guess(p):
    x0 = np.zeros((model.nx,))
    x0[:model.nq] = x0_vec[p]

    ocp.initialize(x0)
    return ocp.getGuess()

def plot_trajectory(x, x_sim):
    fig, ax = plt.subplots(3, 1, sharex='col')
    ax[0].plot(t, x[:-1, 0], label='q1')
    ax[0].plot(t, x_sim[:-1, 0], label='q1_sim', linestyle='--')
    ax[0].legend()
    ax[1].plot(t, x[:-1, 1], label='q2')
    ax[1].plot(t, x_sim[:-1, 1], label='q2_sim', linestyle='--')
    ax[1].legend()
    ax[2].plot(t, x[:-1, 2], label='q3')
    ax[2].plot(t, x_sim[:-1, 2], label='q3_sim', linestyle='--')
    ax[2].legend()
    ax[2].set_xlabel('time [s]')

def simulate(x0, u):
    x_sim = np.empty((conf.N + 1, model.nx)) * np.nan
    x_sim[0] = x0

    for j in range(conf.N):
        x_sim[j + 1] = simulator.simulate(x_sim[j], u[j])
    return x_sim

res = []
for i in range(test_num):
    res.append(init_guess(i))
x_guess_vec, u_guess_vec = zip(*res)

res_sim = []
for i in range(test_num):
    x0 = np.zeros((model.nx,))
    x0[:model.nq] = x_guess_vec[i][0,:3]    #x0_vec[i]
    res_sim.append(simulate(x0, u_guess_vec[i]))
x_sim_vec = np.array(res_sim)

for j in range(test_num):
    plot_trajectory(x_guess_vec[j], x_sim_vec[j])
    # print('i = ', j, ', x0 = ', x0_vec[j])
    plt.savefig('plots/test_integration' + str(j) + '.png')
    plt.close()




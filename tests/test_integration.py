import numpy as np
from scipy.stats import qmc
from safe_mpc.parser import Parameters
from safe_mpc.model import TriplePendulumModel
from safe_mpc.abstract import SimDynamics
from safe_mpc.controller import NaiveController
from debug import Debug

conf = Parameters('../config/params.yaml')
model = TriplePendulumModel(conf)
simulator = SimDynamics(model)
ocp = NaiveController(simulator)
debugger = Debug(conf, ocp)

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


def simulate(x0, u):
    x_sim = np.empty((conf.n_steps + 1, model.nx)) * np.nan
    x_sim[0, :3] = x0
    x_sim[0, 3:] = np.zeros(3)

    for k in range(conf.N):
        x_sim[k + 1] = np.copy(simulator.simulate(x_sim[k], u[k]))
    return x_sim


x_guess_vec, u_guess_vec = [], []
for i in range(test_num):
    xg, ug = init_guess(i)
    x_guess_vec.append(xg)
    u_guess_vec.append(ug)

for j in range(test_num):
    x_roll = simulate(x0_vec[j], u_guess_vec[j])
    debugger.plotTrajectory(j, x0_vec[j], x_guess_vec[j], x_roll)
    # print('Test number: ', j, ', Diff norm ', np.linalg.norm(x_roll[:conf.N+1] - x_guess_vec[j]))

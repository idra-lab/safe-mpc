# import time
# import pickle
import numpy as np
from scipy.stats import qmc
from safe_mpc.parser import Parameters
from safe_mpc.model import TriplePendulumModel
from safe_mpc.abstract import SimDynamics
from safe_mpc.controller import NaiveController
from debug import Debug

conf = Parameters('../config/params.yaml')
model = TriplePendulumModel(conf)
simulator = SimDynamics(conf, model)
ocp = NaiveController(conf, model, simulator)

t_g = np.arange(0, conf.T, conf.dt)
t_s = np.arange(0, conf.dt * conf.n_steps, conf.dt)

# data_no = pickle.load(open(conf.DATA_DIR + "results_no_constraint.pkl", 'rb'))

sampler = qmc.Halton(d=ocp.ocp.dims.nu, scramble=False)
sample = sampler.random(n=conf.test_num)
eps = 1e-5
l_bounds = model.x_min[:model.nq] + eps
u_bounds = model.x_max[:model.nq] - eps
x0_vec = qmc.scale(sample, l_bounds, u_bounds)
# x0_vec = data_no['x0_vec']

x_ref = np.array([conf.q_max - 0.05, np.pi, np.pi, 0, 0, 0])
ocp.setReference(x_ref)


def init_guess(p):
    x0 = np.zeros((model.nx,))
    x0[:model.nq] = x0_vec[p]

    ocp.initialize(x0)
    return ocp.getGuess()


def simulate(p):
    x0 = np.zeros((model.nx,))
    x0[:model.nq] = x0_vec[p]

    x_sim = np.empty((conf.n_steps + 1, model.nx)) * np.nan
    u = np.empty((conf.n_steps, model.nu)) * np.nan
    x_sim[0] = x0

    controller.setGuess(x_guess_vec[p], u_guess_vec[p])

    j = 0
    for j in range(conf.n_steps):
        u[j] = controller.step(x_sim[j])
        x_sim[j + 1] = simulator.simulate(x_sim[j], u[j])
        if not model.checkStateConstraints(x_sim[j + 1]):
            break

    debugger.plotTrajectory(p, x0_vec[p], x_guess_vec[p], x_sim)
    return j, x_sim


x_guess_vec, u_guess_vec = [], []
for i in range(conf.test_num):
    xg, ug = init_guess(i)
    x_guess_vec.append(xg)
    u_guess_vec.append(ug)

print('Init guess success: ' + str(ocp.success) + ' over ' + str(conf.test_num))

del ocp
conf.solver_type = 'SQP_RTI'
controller = NaiveController(conf, model, simulator)
debugger = Debug(conf, controller)
controller.setReference(x_ref)

res = []
for i in range(conf.test_num):
    res.append(simulate(i))
res_steps, x_sim_vec = zip(*res)
res_steps = np.array(res_steps)

print('Residual steps:\n', res_steps)
print('Mean residual steps: ', np.mean(res_steps))

print(np.where(res_steps < controller.N)[0].size)

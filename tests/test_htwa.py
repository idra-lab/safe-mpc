import numpy as np
from scipy.stats import qmc
from safe_mpc.parser import Parameters
from safe_mpc.model import TriplePendulumModel
from safe_mpc.abstract import SimDynamics
from safe_mpc.controller import HTWAController
from debug import Debug

conf = Parameters('../config/params.yaml')
model = TriplePendulumModel(conf)
simulator = SimDynamics(model)
ocp = HTWAController(simulator)
ocp.ocp_solver.set(conf.N, "p", conf.alpha)
debugger = Debug(conf, ocp)

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


x_guess_vec, u_guess_vec = [], []
for i in range(conf.test_num):
    xg, ug = init_guess(i)
    x_guess_vec.append(xg)
    u_guess_vec.append(ug)

for j in range(conf.test_num):
    debugger.plotTrajectory(j, x0_vec[j], x_guess_vec[j])

print('Init guess success: ' + str(ocp.success) + ' over ' + str(conf.test_num))

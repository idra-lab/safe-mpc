import numpy as np
from scipy.stats import qmc
from safe_mpc.parser import Parameters
from safe_mpc.model import TriplePendulumModel
from safe_mpc.abstract import SimDynamics
from safe_mpc.controller import RecedingController


conf = Parameters('triple_pendulum', 'receding', rti=False)
conf.test_num = 500
model = TriplePendulumModel(conf)
simulator = SimDynamics(model)
ocp = RecedingController(simulator)

tol = 1e-3

sampler = qmc.Halton(d=ocp.ocp.dims.nu, scramble=False)
sample = sampler.random(n=conf.test_num)
eps = 1e-5
l_bounds = model.x_min[:model.nq] + eps
u_bounds = model.x_max[:model.nq] - eps
x0_vec = qmc.scale(sample, l_bounds, u_bounds)

x_ref = np.array([conf.q_max - 0.05, np.pi, np.pi, 0, 0, 0])
ocp.setReference(x_ref)

for i in range(conf.N):
    ocp.ocp_solver.cost_set(i, "zl", conf.ws_r * np.ones((1,)))
ocp.ocp_solver.cost_set(conf.N, "zl", conf.ws_t * np.ones((1,)))


def init_guess(p):
    x0 = np.zeros((model.nx,))
    x0[:model.nq] = x0_vec[p]

    flag = ocp.initialize(x0)
    return flag, ocp.getGuess()


success = 0
flags, x_guess_vec, u_guess_vec = [], [], []
for i in range(conf.test_num):
    flag, (xg, ug) = init_guess(i)
    x_guess_vec.append(xg)
    u_guess_vec.append(ug)
    flags.append(flag)
    success += flag

idx = np.where(np.asarray(flags) == 1)[0]

print('Init guess success: ' + str(success) + ' over ' + str(conf.test_num))
np.save(conf.DATA_DIR + '/x_init.npy', np.asarray(x0_vec)[idx])
# np.save(conf.DATA_DIR + '/initial_conditions/x_guess_vec.npy', np.asarray(x_guess_vec)[idx])
# np.save(conf.DATA_DIR + '/initial_conditions/u_guess_vec.npy', np.asarray(u_guess_vec)[idx])

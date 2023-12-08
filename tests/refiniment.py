import numpy as np
from scipy.stats import qmc
from safe_mpc.parser import Parameters
from safe_mpc.model import TriplePendulumModel
from safe_mpc.abstract import SimDynamics
from safe_mpc.controller import RecedingController


conf = Parameters('../config/params.yaml')
model = TriplePendulumModel(conf)
simulator = SimDynamics(model)
ocp = RecedingController(simulator)

sampler = qmc.Halton(d=ocp.ocp.dims.nu, scramble=False)
sample = sampler.random(n=conf.test_num)
eps = 1e-5
l_bounds = model.x_min[:model.nq] + eps
u_bounds = model.x_max[:model.nq] - eps
x0_vec = np.load(conf.DATA_DIR + '/initial_conditions/x_init.npy')[:conf.test_num]
x_guess_vec = np.load(conf.DATA_DIR + '/initial_conditions/x_guess_vec.npy')[:conf.test_num]
u_guess_vec = np.load(conf.DATA_DIR + '/initial_conditions/u_guess_vec.npy')[:conf.test_num]

for i in range(conf.N):
    ocp.ocp_solver.cost_set(i, "zl", conf.ws_r * np.ones((1,)))
ocp.ocp_solver.cost_set(conf.N, "zl", conf.ws_t * np.ones((1,)))


def init_guess(p):
    f = 0
    x0 = np.zeros((model.nx,))
    x0[:model.nq] = x0_vec[p]

    # Set the already optimal guess
    ocp.setGuess(x_guess_vec[p], u_guess_vec[p])
    status = ocp.solve(x0)
    if status == 0 and ocp.checkGuess():
        f = 1
    else:
        print(p)
    return f, ocp.getGuess()


flags, x_sol_vec, u_sol_vec = [], [], []
for i in range(conf.test_num):
    flag, (xg, ug) = init_guess(i)
    x_sol_vec.append(xg)
    u_sol_vec.append(ug)
    flags.append(flag)

idx = np.where(np.asarray(flags) == 1)[0]
print('Init guess success: ' + str(np.sum(flags)) + ' over ' + str(conf.test_num))

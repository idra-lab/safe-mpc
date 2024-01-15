import numpy as np
from scipy.stats import qmc
from safe_mpc.parser import Parameters
from safe_mpc.model import TriplePendulumModel
from safe_mpc.abstract import SimDynamics, IpoptController
from safe_mpc.gravity_compensation import GravityCompensation


conf = Parameters('triple_pendulum', 'naive')
conf.test_num = 50
model = TriplePendulumModel(conf)
simulator = SimDynamics(model)
gravity = GravityCompensation(conf, model)

tol = 1e-3

sampler = qmc.Halton(d=model.nu, scramble=False)
sample = sampler.random(n=conf.test_num)
eps = 1e-5
l_bounds = model.x_min[:model.nq] + eps
u_bounds = model.x_max[:model.nq] - eps
x0_vec = qmc.scale(sample, l_bounds, u_bounds)

x_ref = np.array([conf.q_max - 0.05, np.pi, np.pi, 0, 0, 0])


def init_guess(p):
    x0 = np.zeros((model.nx,))
    x0[:model.nq] = x0_vec[p]
    # u0 = gravity.solve(x0)

    ocp = IpoptController(conf, model, x_ref)

    return ocp.solve(x0)


flags, x_guess_vec, u_guess_vec = [], [], []
for i in range(conf.test_num):
    flag = init_guess(i)
    flags.append(flag)

idx = np.where(np.asarray(flags) == 1)[0]

print('Init guess success: ' + str(np.sum(flags)) + ' over ' + str(conf.test_num))
# np.save(conf.DATA_DIR + 'x_init.npy', np.asarray(x0_vec)[idx])
# np.save(conf.DATA_DIR + 'x_guess_vec.npy', np.asarray(x_guess_vec)[idx])
# np.save(conf.DATA_DIR + 'u_guess_vec.npy', np.asarray(u_guess_vec)[idx])

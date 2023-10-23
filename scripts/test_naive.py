import time
import pickle
import numpy as np
from scipy.stats import qmc
import pendulum_conf as conf
from safe_mpc.pendulum_model import TriplePendulumModel
from safe_mpc.model import SimDynamics
from safe_mpc.naive import NaiveController


model = TriplePendulumModel(conf)
simulator = SimDynamics(conf, model)
ocp = NaiveController(conf, model)

data_dir = '/home/gianni/devel/src/safe-mpc/data/'
data_no = pickle.load(open(data_dir + "results_no_constraint.pkl", 'rb'))

test_num = 100
sampler = qmc.Halton(d=ocp.ocp.dims.nu, scramble=False)
sample = sampler.random(n=test_num)
l_bounds = model.x_min[:model.nq]
u_bounds = model.x_max[:model.nq]
x0_vec = qmc.scale(sample, l_bounds, u_bounds)
# x0_vec = data_no['x0_vec']

x_ref = np.array([conf.q_max-0.05, np.pi, np.pi, 0, 0, 0])
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

    controller.setGuess(np.copy(x_guess_vec[p]), np.copy(u_guess_vec[p]))

    for j in range(conf.n_steps):
        u[j] = controller.step(x_sim[j])
        x_sim[j+1] = simulator.simulate(x_sim[j], u[j])
        if model.checkStateConstraints(x_sim[j+1]) == False:
            break
    return j

res = []
for i in range(test_num):
    res.append(init_guess(i))
x_guess_vec, u_guess_vec = zip(*res)

print('Init guess success: ' + str(ocp.success) + ' over ' + str(test_num))

del ocp
conf.solver_type = 'SQP_RTI'
controller = NaiveController(conf, model)

res_steps = np.zeros(test_num)
for i in range(test_num):
    res_steps[i] = simulate(i)

print('Residual steps:\n', res_steps)
print('Mean residual steps: ', np.mean(res_steps)) 

x_test = data_no['x_guess_vec']
diff_norm = np.zeros(test_num)
for i in range(test_num): 
    diff_norm[i] = np.linalg.norm(x_guess_vec[i] - x_test[i])

print('diff norm: ', diff_norm)
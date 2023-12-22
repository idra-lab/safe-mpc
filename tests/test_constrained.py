import numpy as np
from safe_mpc.parser import Parameters
from safe_mpc.model import TriplePendulumModel
from safe_mpc.abstract import SimDynamics
from safe_mpc.controller import STWAController, HTWAController, RecedingController
from debug import Debug

conf = Parameters('../config/params.yaml')
conf.solver_type = 'SQP_RTI'
conf.globalization = 'FIXED_STEP'
model = TriplePendulumModel(conf)
simulator = SimDynamics(model)
controller = RecedingController(simulator)
debugger = Debug(conf, controller)
tol = 1e-3

x0_vec = np.load(conf.DATA_DIR + 'x_init.npy')[:conf.test_num]
x_ref = np.array([conf.q_max - 0.05, np.pi, np.pi, 0, 0, 0])
controller.setReference(x_ref)


def simulate(p):
    x0 = np.zeros((model.nx,))
    x0[:model.nq] = x0_vec[p]

    x_sim = np.empty((conf.n_steps + 1, model.nx)) * np.nan
    u = np.empty((conf.n_steps, model.nu)) * np.nan
    x_sim[0] = x0

    controller.setGuess(x_guess_vec[p], u_guess_vec[p])
    controller.fails = 0
    stats = []
    convergence = 0
    j = 0
    for j in range(conf.n_steps):
        u[j] = controller.step(x_sim[j])
        stats.append(controller.getTime())
        x_sim[j + 1] = simulator.simulate(x_sim[j], u[j])
        # Check if the next state is inside the state bounds
        if not model.checkStateConstraints(x_sim[j + 1]):
            break
        # Check convergence --> norm of diff btw x_sim and x_ref (only for first joint)
        if np.linalg.norm([x_sim[j + 1, 0] - x_ref[0], x_sim[j + 1, 3]]) < tol:
            convergence = 1
            break

    x_v = controller.getLastViableState()
    debugger.plotTrajectory(p, x0_vec[p], x_guess_vec[p], x_sim)
    return j, convergence, x_sim, x_v, stats


x_guess_vec = np.load(conf.DATA_DIR + 'x_guess_vec.npy')[:conf.test_num]
u_guess_vec = np.load(conf.DATA_DIR + 'u_guess_vec.npy')[:conf.test_num]

res = []
for i in range(conf.test_num):
    res.append(simulate(i))
steps, conv_vec, x_sim_vec, x_viable, t_stats = zip(*res)
steps = np.array(steps)
conv_vec = np.array(conv_vec)
idx = np.where(conv_vec == 1)[0]
idx_abort = np.where(conv_vec == 0)[0]
times = np.array([t for arr in t_stats for t in arr])

print('Residual steps:\n', steps)
print('Total convergence: ' + str(np.sum(conv_vec)) + ' over ' + str(conf.test_num))

fields = ['time_lin', 'time_sim', 'time_qp', 'time_qp_solver_call',
          'time_glob', 'time_reg', 'time_tot']
print('Mean computation time:')
for field, t in zip(fields, np.mean(times, axis=0)):
    print(f"{field:<20} -> {t[0]}")
print('99% quantile computation time:')
for field, t in zip(fields, np.quantile(times, 0.99, axis=0)):
    print(f"{field:<20} -> {t[0]}")

np.save(conf.DATA_DIR + '/' + controller.ocp_name + '/x_viable.npy', np.asarray(x_viable)[idx_abort])

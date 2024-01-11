import numpy as np
from safe_mpc.parser import Parameters
from safe_mpc.model import TriplePendulumModel
from safe_mpc.abstract import SimDynamics
from safe_mpc.gravity_compensation import GravityCompensation
from safe_mpc.abort import VbocLikeController


def linear_sat(u, u_bar):
    dim = len(u)
    res = np.zeros(dim)
    for i in range(dim):
        if u[i] < - u_bar:
            res[i] = -u_bar
        elif u[i] > u_bar:
            res[i] = u_bar
        else:
            res[i] = u[i]
    return res

def create_guess(x0, target):
    kp = 1e-2 * np.eye(3)
    kd = 1e2 * np.eye(3)
    simX = np.empty((ocp.N + 1, ocp.ocp.dims.nx)) * np.nan
    simU = np.empty((ocp.N, ocp.ocp.dims.nu)) * np.nan
    simX[0] = np.copy(x0)
    try:
        u0 = gravity.solve(x0)
    except:
        u0 = np.zeros((model.nu,))
    for i in range(ocp.N):
        simU[i] = linear_sat(kp.dot(target[:3] - simX[i,:3]) + kd.dot(target[3:] - simX[i,3:]), conf.u_max) + u0
        simX[i + 1] = sim.simulate(simX[i], simU[i])
    # Then saturate the state
    for i in range(ocp.N):
        simX[i,:3] = linear_sat(simX[i,:3], conf.q_max)
        simX[i,3:] = linear_sat(simX[i,3:], conf.dq_max)
    return simX, simU

conf = Parameters('../config/params.yaml')
conf.T = 0.2
model = TriplePendulumModel(conf)
sim = SimDynamics(model)
gravity = GravityCompensation(conf, model)
ocp = VbocLikeController(conf, model)

controller_name = 'receding'        # 'stwa', 'htwa', 'receding'
x_viable = np.load(conf.DATA_DIR + '/' + controller_name + '/x_viable.npy')

n = x_viable.shape[0]
success = np.zeros(n)
real_succ = 0                       # from the solved problems, find the ones for which the initial velocity is 
                                    # sufficiently close to the real one
t_comp = []
for i in range(n):
    x0 = np.copy(x_viable[i])

    x_ref = np.array([np.pi, np.pi, np.pi, 0., 0., 0.])
    x_guess, u_guess = create_guess(x0, x_ref)

    # status = ocp.solve(x0, np.full((conf.N + 1, model.nx), x0), np.zeros((conf.N, model.nu)))
    status = ocp.solve(x0, x_guess, u_guess)
    # ocp.ocp_solver.print_statistics()
    print(status)
    if status == 0:
        t_comp.append(ocp.ocp_solver.get_stats('time_tot')[0])
        success[i] = 1

        norm_init = np.linalg.norm(x0[model.nq:])
        norm_sol = np.linalg.norm(ocp.ocp_solver.get(0, "x")[model.nq:])
        diff = norm_init - norm_sol
        perc = diff / norm_init * 100
        print(perc)
        if perc < 10:
            real_succ += 1

print('Controller: ' + controller_name)
print('Success: ' + str(np.sum(success)) + ' over ' + str(n))
print('Abort: ' + str(real_succ) + ' over ' + str(n))
print('99% quantile time: ', np.quantile(np.asarray(t_comp), 0.99))
import numpy as np
from safe_mpc.parser import Parameters
from safe_mpc.model import TriplePendulumModel
from safe_mpc.abstract import SimDynamics
from safe_mpc.abort import SafeBackupController


def evaluate_nn(x):
    n = np.shape(x)[0]
    out_vec = np.zeros((n,))
    for h in range(n):
        out_vec[h] = model.nn_func(x[h], conf.alpha)
    return out_vec

conf = Parameters('../config/params.yaml')
conf.T = 0.2
model = TriplePendulumModel(conf)
model.setNNmodel()
simulator = SimDynamics(model)
ocp = SafeBackupController(simulator)

x_viable = np.load(conf.DATA_DIR + '/viable/x_viable.npy')
nn_out = evaluate_nn(x_viable)
print(nn_out)

n = x_viable.shape[0]
success = np.zeros(n)
t_comp = []
for i in range(n):
    ocp.setGuess(np.full((conf.N + 1, model.nx), x_viable[i]), np.zeros((conf.N, model.nu)))
    status = ocp.solve(x_viable[i])
    print(status)
    if status == 0:
        t_comp.append(ocp.ocp_solver.get_stats('time_tot')[0])
        success[i] = 1

print('Abort: ' + str(np.sum(success)) + ' over ' + str(n))
print('99% quantile time: ', np.quantile(np.asarray(t_comp), 0.99))

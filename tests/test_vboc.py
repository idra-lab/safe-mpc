import numpy as np
from safe_mpc.parser import Parameters
from safe_mpc.model import TriplePendulumModel
from safe_mpc.abort import VbocLikeController

conf = Parameters('../config/params.yaml')
conf.T = 0.2
model = TriplePendulumModel(conf)
ocp = VbocLikeController(conf, model)

controller_name = 'receding'        # 'stwa', 'htwa', 'receding'
x_viable = np.load(conf.DATA_DIR + '/' + controller_name + '/x_viable.npy')

n = x_viable.shape[0]
success = np.zeros(n)
t_comp = []
for i in range(n):
    x0 = np.copy(x_viable[i])
    status = ocp.solve(x0, np.full((conf.N + 1, model.nx), x0), np.zeros((conf.N, model.nu)))
    print(status)
    if status == 0:
        t_comp.append(ocp.ocp_solver.get_stats('time_tot')[0])
        success[i] = 1

print('Controller: ' + controller_name)
print('Abort: ' + str(np.sum(success)) + ' over ' + str(n))
print('99% quantile time: ', np.quantile(np.asarray(t_comp), 0.99))
import pickle
import numpy as np
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel
from safe_mpc.controller import NaiveController
from safe_mpc.ocp import NaiveOCP
from safe_mpc.cost_definition import *

def open_pickle(name):
    """ Name = model_name + cont_name """
    return pickle.load(open(f'{params.DATA_DIR}{name}_mpc.pkl', 'rb'))

def cost(x, u):
    n = len(u)
    cost = 0
    for i in range(n):
        delta_ee = model.jointToEE(x[i]) - params.ee_ref.reshape(3, 1)
        cost += delta_ee.T @ Q @ delta_ee
        cost += u[i].dot(R.dot(u[i]))
    delta_ee = model.jointToEE(x[-1]) - params.ee_ref.reshape(3, 1)
    cost += delta_ee.T @ Q @ delta_ee
    return cost[0, 0]

COMPUTE_OPT_TRAJ = False
WARM_START = False

args = parse_args()
model_name = args['system']
hor = args['horizon']
alpha = args['alpha']
params = Parameters(model_name, rti=True)
model = AdamModel(params)

if WARM_START:
    data = pickle.load(open(f'{params.DATA_DIR}{model_name}_receding_use_netTrue_{hor}hor_{int(alpha)}sm_mpc.pkl', 'rb'))
    X_traj = data['x']
    U_traj = data['u']

controller = NaiveController(model)
cost_controller = ReachTargetNLS(model,params.Q_weight,params.R_weight)
cost_controller.set_solver_cost(controller)

Q, R = params.Q_weight*np.eye(controller.model.t_glob.shape[0]), params.R_weight*np.eye(controller.model.nu)

params.N = params.n_steps
ocp = NaiveOCP(model)

opti = ocp.opti
opts = {
    'ipopt.print_level': 2,
    'print_time': 0,
    'ipopt.tol': 1e-6,
    'ipopt.constr_viol_tol': 1e-6,
    'ipopt.compl_inf_tol': 1e-6,
    'ipopt.max_iter': 400
    #'ipopt.linear_solver': 'ma57'
}
opti.solver('ipopt', opts)
costs_list = []

cont_names = ['naive', 'zerovel', 'st', 'htwa', 'receding','real_receding','parallel']

x_init = {}

x_opt_list = []

for c in cont_names:
    if c in ['st', 'htwa', 'receding','real_receding','parallel']:
        #print(c)
        use_net = True
    else:
        use_net = None
    x_init[c] = []
    data = pickle.load(open(f'{params.DATA_DIR}{model_name}_{c}_{hor}hor_{int(params.alpha)}sm_use_net{use_net}__guess.pkl', 'rb'))['xg']
    for j in range(data.shape[0]):
        x_init[c].append(data[j,0,:])

x_guess = pickle.load(open(f'{params.DATA_DIR}{model_name}_{cont_names[0]}_{hor}hor_{int(params.alpha)}sm_use_net{None}__guess.pkl', 'rb'))['xg']
u_guess = pickle.load(open(f'{params.DATA_DIR}{model_name}_{cont_names[0]}_{hor}hor_{int(params.alpha)}sm_use_net{None}__guess.pkl', 'rb'))['ug']

opt_scores = []

all_eq = True
for i in range(1,len(x_init[cont_names[0]])):
    for k in range(0,len(cont_names)):
        all_eq = all_eq and (np.linalg.norm(x_init[cont_names[0]][i] - x_init[cont_names[k]][i])<1e-2)
        if not(all_eq):
            print(f'different states at index {i}')
            exit()

for i in range(len(x_init[cont_names[0]])):
    print(f'Problem number {i + 1}/{params.test_num}')
        
    # Use the controller with lowest score as initial guess

    if WARM_START:
        opti.set_value(ocp.x_init, X_traj[i,0,:])
        for k in range(1,params.n_steps+1):
            if not(np.isnan(X_traj[i,k,:]).all()):
                opti.set_initial(ocp.X[k], X_traj[i,k,:])
        for k in range(params.n_steps):
            if not(np.isnan(U_traj[i,k,:]).all()):
                opti.set_initial(ocp.U[k], U_traj[i,k,:])
    else:
        opti.set_value(ocp.x_init, x_init[cont_names[0]][i])
        for k in range(params.n_steps):
            opti.set_initial(ocp.X[k], x_init[cont_names[0]][i])
        opti.set_initial(ocp.X[-1], x_init[cont_names[0]][i])

    try:
        sol = opti.solve()
        x_opt = np.array([sol.value(ocp.X[k]) for k in range(params.n_steps + 1)])
        u_opt = np.array([sol.value(ocp.U[k]) for k in range(params.n_steps)])
        cost_opt = cost(x_opt, u_opt)
        print(f'Cost: {cost_opt}')
        opt_scores.append(cost_opt)    

        x_opt_list.append(x_opt)
    except:
        print(f'\t\tFailed to solve the optimal trajectory, use {1e10} score')
        opt_scores.append(1e10)

    if i % 10 == 0:
        with open(f'{params.DATA_DIR}x_traj_opt.pkl', 'wb') as f:
            pickle.dump(x_opt_list,f)

np.save(f'{params.DATA_DIR}{model_name}_{hor}hor_{int(alpha)}sm_opt_costs.npy', np.asarray(opt_scores))
import pickle
import numpy as np
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel
from safe_mpc.utils import obstacles, ee_ref
from safe_mpc.controller import NaiveController
from safe_mpc.ocp import NaiveOCP


def open_pickle(name):
    """ Name = model_name + cont_name """
    return pickle.load(open(f'{params.DATA_DIR}{name}_mpc.pkl', 'rb'))

def cost(x, u):
    n = len(u)
    cost = 0
    for i in range(n):
        delta_ee = model.jointToEE(x[i]) - ee_ref.reshape(3, 1)
        cost += delta_ee.T @ Q @ delta_ee
        cost += u[i].dot(R.dot(u[i]))
    delta_ee = model.jointToEE(x[-1]) - ee_ref.reshape(3, 1)
    cost += delta_ee.T @ Q @ delta_ee
    return cost[0, 0]


COMPUTE_OPT_TRAJ = False

args = parse_args()
model_name = args['system']
hor = args['horizon']
alpha = args['alpha']
params = Parameters(model_name, rti=True)
model = AdamModel(params, n_dofs=4)
model.ee_ref = ee_ref

controller = NaiveController(model, obstacles)
Q, R = controller.Q, controller.R

if COMPUTE_OPT_TRAJ:
    params.N = params.n_steps
    ocp = NaiveOCP(model, obstacles)
    opti = ocp.opti
    opts = {
        'ipopt.print_level': 5,
        'print_time': 0,
        'ipopt.tol': 1e-6,
        'ipopt.constr_viol_tol': 1e-6,
        'ipopt.compl_inf_tol': 1e-6,
        'ipopt.max_iter': 200,
        'ipopt.linear_solver': 'ma57'
    }
    opti.solver('ipopt', opts)
    costs_list = []
else:
    costs_list = np.load(f'{params.DATA_DIR}{model_name}_opt_costs.npy')  

cont_names = ['naive', 'zerovel', 'st', 'terminal', 'htwa', 'receding']
cont_names = cont_names[2:]           # This for the multiple alpha case
X_traj, U_traj, task_not_coll, task_failed = {}, {}, {}, {}
for c in cont_names:
    data = open_pickle(f'{model_name}_{c}_{hor}hor_{int(alpha)}sm')
    X_traj[c] = data['x']
    U_traj[c] = data['u']
    task_not_coll[c] = np.union1d(data['conv_idx'], data['unconv_idx'])
    task_failed[c] = data['collisions_idx']

tot_scores = {c: [] for c in cont_names}

for i in range(params.test_num):
    print(f'Problem number {i + 1}/{params.test_num}')
    not_coll = [True] * len(cont_names)
    for j, c in enumerate(cont_names):
        if i in task_failed[c]:
            not_coll[j] = False
            break

    costs = np.array([cost(X_traj[c][i], U_traj[c][i]) for c in cont_names])
    if COMPUTE_OPT_TRAJ:
        try:
            j_min = np.nanargmin(costs)
            c_min = cont_names[j_min]
        except:
            print('\tAll controllers failed to solve the task')
            continue
        
        # Use the controller with lowest score as initial guess
        opti.set_value(ocp.x_init, X_traj[c_min][i][0])
        for k in range(params.n_steps):
            opti.set_initial(ocp.X[k], X_traj[c_min][i][k])
            opti.set_initial(ocp.U[k], U_traj[c_min][i][k])
        opti.set_initial(ocp.X[-1], X_traj[c_min][i][-1])

        try:
            sol = opti.solve()
            x_opt = np.array([sol.value(ocp.X[k]) for k in range(params.n_steps + 1)])
            u_opt = np.array([sol.value(ocp.U[k]) for k in range(params.n_steps)])
            cost_opt = cost(x_opt, u_opt)
            if cost_opt < costs[j_min]:
                min_cost = cost_opt
            else:
                print(f'\t\tFailed to solve the optimal trajectory, use {c_min} score')
                min_cost = costs[j_min]
        except:
            print(f'\t\tFailed to solve the optimal trajectory, use {c_min} score')
            min_cost = costs[j_min]
        costs_list.append(min_cost)
    
    else:
        min_cost = costs_list[i]

    if all(not_coll):
        print('\tAll controllers solved the task')

        norm_score = costs / min_cost
        for j, c in enumerate(cont_names):
            print(f'{c} -> {norm_score[j]}', end=' ')
            tot_scores[c] += [norm_score[j]]
        print()

    else:
        print('\tSome controllers failed to solve the task')
        for j, c in enumerate(cont_names):
            if not_coll[j]:
                print(f'\t{c} solved the task')
            else:
                print(f'\t{c} failed to solve the task')

if COMPUTE_OPT_TRAJ:
    np.save(f'{params.DATA_DIR}{model_name}_opt_costs.npy', np.asarray(costs_list))

print('\n### Final scores: ###\n')
res = {}
for c in cont_names:
    print(f'Controller: {c}')
    print(f'Number of scores -> {len(tot_scores[c])}')
    mean_score = np.mean(tot_scores[c])
    perc_score = round((mean_score - 1) * 100, 2)
    print(f'Mean of scores -> {mean_score}, perc -> {perc_score} \n')

    res[c] = {}
    res[c]['score'] = perc_score
    res[c]['fails'] = len(task_failed[c])

file_prefix = f'{params.DATA_DIR}{model_name}_{hor}hor_{int(alpha)}sm'
with open(f'{file_prefix}_scores.pkl', 'wb') as f:
    pickle.dump(res, f) 
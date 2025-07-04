import pickle
import numpy as np
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel
from safe_mpc.controller import NaiveController
from safe_mpc.ocp import NaiveOCP
import os

def find_cost(x0):
    for indx,item in enumerate(costs_states['states']):
        if (np.abs(item - x0) < 1e-3).all():
            return indx
    return False

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

args = parse_args()
model_name = args['system']
hor = args['horizon']
alpha = args['alpha']
params = Parameters(args, model_name, rti=True)
model = AdamModel(params)
noise = args['noise']

controller = NaiveController(model)
Q, R = params.Q_weight*np.eye(controller.model.t_glob.shape[0]), params.R_weight*np.eye(controller.model.nu)

traj__track = ''

# check if file of optimal costs - initial state exists
file = f'{params.DATA_DIR}costs_state.pkl'

if os.path.isfile(file):
    costs_states = pickle.load(open(file,'rb'))
else:
    costs_states = {}
    costs_states['states'] = []
    costs_states['costs'] = []

cont_names = ['naive', 'zerovel', 'st', 'htwa', 'receding','parallel2','receding_parallel','constraint_everywhere']

X_traj, U_traj, task_not_coll, task_failed = {}, {}, {}, {}
for c in cont_names:
    if c in [ 'st', 'htwa', 'receding','parallel2','receding_parallel','constraint_everywhere']:
        #print(c)
        use_net = True
    else:
        use_net = None
    try:
        data = pickle.load(open(f'{params.DATA_DIR}{model_name}_{c}_use_net{use_net}_{hor}hor_{int(args["alpha"])}sm_{traj__track}noise_{args["noise"]}_control_noise{args["control_noise"]}_q_collision_margins_{args["joint_bounds_margin"]}_{args["collision_margin"]}_mpc.pkl', 'rb'))
        X_traj[c] = data['x']
        U_traj[c] = data['u']
        task_not_coll[c] = np.union1d(data['conv_idx'], data['unconv_idx'])
        task_failed[c] = data['collisions_idx']
    except:
        task_not_coll[c] = [100]*100
        task_failed[c] = []

print('\n### Final scores: ###\n')
res = {}
for c in cont_names:
    res[c] = {}
    res[c]['score'] = 0
    res[c]['fails'] = len(task_failed[c])

file_prefix = f'{params.DATA_DIR}{model_name}_{hor}hor_{int(alpha)}sm_noise{noise}_control_noise{args["control_noise"]}_q_collision_margins_{args["joint_bounds_margin"]}_{args["collision_margin"]}'
with open(f'{file_prefix}_scores.pkl', 'wb') as f:
    pickle.dump(res, f) 

print(f"scores {res['htwa']['fails']}")


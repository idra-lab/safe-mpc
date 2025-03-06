import time
import pickle
import numpy as np
from scipy.stats import qmc
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel
from safe_mpc.utils import get_ocp, RobotVisualizer
from tqdm import tqdm

args = parse_args()
model_name = args['system']
params = Parameters(model_name, rti=False)
params.N = args['horizon']
params.alpha = args['alpha']
model = AdamModel(params)

ocp_names = ['naive','zerovel','st','htwa','receding','parallel','real']
ocps, optis = [],[]
for name in ocp_names:
    ocps.append(get_ocp(name, model))
    optis.append(ocps[-1].instantiateProblem())

num_ics = params.test_num
succ, fails, skip_ics = 0, 0, 0
sampler = qmc.Halton(model.nq, scramble=False)
x_guess, u_guess = [[] for _ in ocp_names], [[] for _ in ocp_names]

# rviz = RobotVisualizer(params, n_dofs=4)
# if params.obs_flag:
#     rviz.addObstacles(params.obstacles)
# rviz.init_capsule(params.robot_capsules+params.obst_capsules)

start_time = time.time()
progress_bar = tqdm(total=num_ics, desc=f'Generating initial conditions, alpha {ocps[0].model.params.alpha}')
while succ < num_ics:
    q0 = qmc.scale(sampler.random(), model.x_min[:model.nq], model.x_max[:model.nq])[0]
    x0 = np.zeros((model.nx,))
    x0[:model.nq] = q0

    #rviz.displayWithEESphere(x0[:4],params.robot_capsules+params.obst_capsules)

    if ocps[0].model.checkCollision(x0):
        u0 = np.zeros((model.nu,)) 

        for i,(ocp,opti) in enumerate(zip(ocps,optis)):
            opti.set_value(ocp.x_init, x0)
            for k in range(params.N):
                opti.set_initial(ocp.X[k], x0)
                opti.set_initial(ocp.U[k], u0)
            opti.set_initial(ocp.X[-1], x0)
        try:
            sol = opti.solve()
            xg = np.array([sol.value(ocp.X[k]) for k in range(params.N + 1)])
            ug = np.array([sol.value(ocp.U[k]) for k in range(params.N)])
            x_guess[i].append(xg), u_guess[i].append(ug)
            succ += 1
            progress_bar.update(1)
        except:
            sol = opti.debug
            fails += 1
            print('failed')
            continue
    else:
        print('skipped')
        skip_ics += 1
progress_bar.close()

equal = True
for i in range(len(x_guess[0])):
    for j in range(len(x_guess)):
        if x_guess[0][i][0] != x_guess[j][i][0]:
            equal = False
            break
if equal: 
    print('All initial conditions equal') 
else: 
    print(f'Initial conditions different at index {i} for controller {ocp_names[j]}') 
    exit()

print(f'Number of failed initializations: {fails}')
print(f'Number of skipped initial conditions: {skip_ics}')

traj__track = 'traj_track' if ocp.model.params.track_traj else "" 

for i,controller_name in enumerate(ocp_names):
    x_guess_to_save = np.asarray(x_guess[i])
    u_guess_to_save = np.asarray(u_guess[i])

    with open(f'{params.DATA_DIR}{model_name}_{controller_name}_{args["horizon"]}hor_{int(params.alpha)}sm_use_net{ocp.model.params.use_net}_{traj__track}_guess.pkl', 'wb') as f:
                pickle.dump({'xg': np.asarray(x_guess_to_save), 'ug': np.asarray(u_guess_to_save)}, f)

elapsed_time = time.time() - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)
print(f'Elapsed time: {hours}:{minutes:2d}:{seconds:2d}')

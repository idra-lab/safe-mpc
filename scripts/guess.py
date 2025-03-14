import time
import pickle
import numpy as np
from scipy.stats import qmc
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel
from safe_mpc.utils import get_ocp
from safe_mpc.robot_visualizer import RobotVisualizer
from tqdm import tqdm
from copy import deepcopy

args = parse_args()
model_name = args['system']

ocp_names = ['naive','zerovel','st','htwa','receding','parallel','real']
params_list, models, ocps, optis = [], [], [],[]
for name in ocp_names:
    params_list.append(Parameters(model_name, rti=False))
    params_list[-1].N = args['horizon']
    params_list[-1].alpha = args['alpha']
    models.append(AdamModel(params_list[-1]))
    ocps.append(get_ocp(name, models[-1]))
    optis.append(ocps[-1].instantiateProblem())

num_ics = params_list[0].test_num
succ, fails, skip_ics = 0, 0, 0
sampler = qmc.Halton(models[0].nq, scramble=False)
x_guess, u_guess = [[] for _ in ocp_names], [[] for _ in ocp_names]

rviz = RobotVisualizer(params_list[0], n_dofs=4)
if params_list[0].obs_flag:
    rviz.addObstacles(params_list[0].obstacles)
rviz.init_capsule(params_list[0].robot_capsules+params_list[0].obst_capsules)

start_time = time.time()
progress_bar = tqdm(total=num_ics, desc=f'Generating initial conditions, alpha {ocps[0].model.params.alpha}')
while succ < num_ics:
    q0 = qmc.scale(sampler.random(), models[0].x_min[:models[0].nq], models[0].x_max[:models[0].nq])[0]
    x0 = np.zeros((models[0].nx,))
    x0[:models[0].nq] = q0

    rviz.displayWithEESphere(x0[:4],params_list[0].robot_capsules+params_list[0].obst_capsules, params_list[0].spheres_robot)

    if ocps[0].model.checkCollision(x0):
        u0 = np.zeros((models[0].nu,)) 

        for i,(ocp,opti) in enumerate(zip(ocps,optis)):
            opti.set_value(ocp.x_init, x0)
            for k in range(params_list[i].N):
                opti.set_initial(ocp.X[k], x0)
                opti.set_initial(ocp.U[k], u0)
            opti.set_initial(ocp.X[-1], x0)
            try:
                sol = opti.solve()
                xg = np.array([sol.value(ocp.X[k]) for k in range(params_list[i].N + 1)])
                ug = np.array([sol.value(ocp.U[k]) for k in range(params_list[i].N)])
                x_guess[i].append(deepcopy(xg)), u_guess[i].append(deepcopy(ug))
            except:
                sol = opti.debug
                fails += 1
                print('failed')
                for jj in range(i):
                    del x_guess[jj][-1]
                    del u_guess[jj][-1]    

                break
            if i == len(ocp_names)-1:
                progress_bar.update(1)
                succ += 1
            
    else:
        print('skipped')
        time.sleep(4)
        skip_ics += 1
progress_bar.close()

equal = True
for i in range(len(x_guess[0])):
    for j in range(len(x_guess)):
        if (np.abs(x_guess[0][i][0] - x_guess[j][i][0]) > 1e-4 ).any():
            print(f'j:{j} , i:{i}')
            equal = False
            break
if equal: 
    print('All initial conditions equal') 
else: 
    print(f'Initial conditions different at index {i} for controller {ocp_names[j]}') 
    exit()

print(f'Number of failed initializations: {fails}')
print(f'Number of skipped initial conditions: {skip_ics}')

traj__track = 'traj_track' if ocps[0].model.params.track_traj else "" 

for i,controller_name in enumerate(ocp_names):
    x_guess_to_save = np.asarray(x_guess[i])
    u_guess_to_save = np.asarray(u_guess[i])

    with open(f'{params_list[i].DATA_DIR}{model_name}_{controller_name}_{args["horizon"]}hor_{int(params_list[i].alpha)}sm_use_net{ocps[i].model.params.use_net}_{traj__track}_guess.pkl', 'wb') as f:
                pickle.dump({'xg': np.asarray(x_guess_to_save), 'ug': np.asarray(u_guess_to_save)}, f)

elapsed_time = time.time() - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)
print(f'Elapsed time: {hours}:{minutes:2d}:{seconds:2d}')

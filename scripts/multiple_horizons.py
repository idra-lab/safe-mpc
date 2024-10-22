import os
import pickle
import numpy as np
from tqdm import tqdm
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel
from safe_mpc.utils import obstacles, ee_ref, get_ocp, get_controller


args = parse_args()
model_name = args['system']
params = Parameters(model_name, rti=True)
params.build = args['build']
params.act = args['activation']
model = AdamModel(params, n_dofs=4)
model.ee_ref = ee_ref

ocp_name = args['controller']
ocp_class = get_ocp(ocp_name, model, obstacles).__class__

cont_name = ocp_name
controller = get_controller(cont_name, model, obstacles)
controller.setReference(ee_ref)

data = pickle.load(open(f'{params.DATA_DIR}{model_name}_{cont_name}_guess.pkl', 'rb'))
xg_list = data['xg']
ug_list = data['ug']
x_init = xg_list[:,0,:]
data_dir = params.DATA_DIR + 'mh/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

N_init = np.copy(params.N)
N_last = 20
delta = 5

GUESS_FLAG = True
MPC_FLAG = True


### GUESS ###

if GUESS_FLAG:
    # Generate all the guess for different horizon
    x_tot, u_tot = [], []                   # Store all the guess
    for i in tqdm(range(params.test_num), desc='Initial condition'):
        x0 = x_init[i]
        xg = xg_list[i]
        ug = ug_list[i]

        # Store the guess for different horizons, same IC
        x_hor, u_hor = [], []               
        # for j in range(N_init, N_last - 1, -1):
        for j in tqdm(range(N_init, N_last - 1, -1), desc='Horizon reduction', leave=False):
            params.N = j
            ocp = ocp_class(model, obstacles)
            opti = ocp.instantiateProblem()

            opti.set_value(ocp.x_init, x0)
            for k in range(j):
                opti.set_initial(ocp.X[k], xg[k])
                opti.set_initial(ocp.U[k], ug[k])
            opti.set_initial(ocp.X[-1], xg[j])
            
            try:
                sol = opti.solve()
                xg = np.array([sol.value(ocp.X[k]) for k in range(j + 1)])
                ug = np.array([sol.value(ocp.U[k]) for k in range(j)])
            except:
                print(f'IC: {i}, Horizon: {j}, Failed solution')
            
                # print(model.checkRunningConstraints(controller.x_temp, controller.u_temp), \
                #   model.checkDynamicsConstraints(controller.x_temp, controller.u_temp), \
                #     np.all([controller.checkCollision(x) for x in controller.x_temp]))
                # for i, x in enumerate(controller.x_temp):
                #     t_glob = model.jointToEE(x)
                #     print(f'Iter {i} : ', end='')
                #     no_coll = True
                #     for obs in obstacles:
                #         if obs['name'] == 'floor':
                #             diff = t_glob[2] - obs['bounds'][0]
                #             if diff < 0:
                #                 print(f'Floor penetration : {diff}')
                #                 no_coll = False
                #         elif obs['name'] == 'ball':
                #             dist = np.sum((t_glob.flatten() - obs['position']) ** 2)
                #             diff = dist - obs['bounds'][0]
                #             if diff < 0:
                #                 print(f'Ball penetration : {diff}')
                #                 no_coll = False
                #     if no_coll:
                #         print('No collision')

            # Save the guess for the intesting horizon
            if j % delta == 0:
                x_hor.append(xg), u_hor.append(ug)

        # Save guess for all horizons
        x_tot.append(x_hor), u_tot.append(u_hor)

    # Extract and save the guess in different pickels for each horizon
    for i, j in enumerate(range(N_init, N_last - 1, -delta)):
        x_guess = np.asarray([x[i] for x in x_tot])
        u_guess = np.asarray([u[i] for u in u_tot])
        with open(f'{data_dir}{cont_name}_{j}hor_guess.pkl', 'wb') as f:
            pickle.dump({'xg': x_guess, 'ug': u_guess}, f)


### MPC ###

if MPC_FLAG:
    for k in range(N_init, N_last - 1, -delta):
        data = pickle.load(open(f'{data_dir}{cont_name}_{k}hor_guess.pkl', 'rb'))
        x_guess = data['xg']
        u_guess = data['ug']
        x_init = x_guess[:,0,:]

        # MPC simulation 
        conv, collisions = 0, 0
        controller.resetHorizon(k)
        for i in tqdm(range(params.test_num), desc='MPC simulations', leave=False):
            x0 = x_init[i]
            x_sim = np.empty((params.n_steps + 1, model.nx)) * np.nan
            u = np.empty((params.n_steps, model.nu)) * np.nan
            x_sim[0] = x0

            controller.setGuess(x_guess[i], u_guess[i])
            try:
                controller.r = controller.N
            except:
                pass
            controller.fails = 0
            j = 0
            for j in range(params.n_steps):
                u[j] = controller.step(x_sim[j])
                x_sim[j + 1], _ = model.integrate(x_sim[j], u[j])
                # Check next state bounds and collision
                if not model.checkStateConstraints(x_sim[j + 1]):
                    collisions += 1
                    break
                if not controller.checkCollision(x_sim[j + 1]):
                    collisions += 1
                    break
            # Check convergence
            if np.linalg.norm(model.jointToEE(x_sim[-1]).T - ee_ref) < params.tol_conv:
                conv += 1

        print(f'Horizon: {k}, Completed task: {conv}, Collisions: {collisions}')
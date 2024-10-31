import os
import pickle
import numpy as np
from tqdm import tqdm
from functools import reduce
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

GUESS_FLAG = False
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

            # Save the guess for the intesting horizon
            if j % delta == 0:
                x_hor.append(xg), u_hor.append(ug)

        # Save guess for all horizons
        x_tot.append(x_hor), u_tot.append(u_hor)

    # Extract and save the guess in different pickels for each horizon
    for i, j in enumerate(range(N_init, N_last - 1, -delta)):
        x_guess = np.asarray([x[i] for x in x_tot])
        u_guess = np.asarray([u[i] for u in u_tot])
        with open(f'{data_dir}{model_name}_{cont_name}_{j}hor_guess.pkl', 'wb') as f:
            pickle.dump({'xg': x_guess, 'ug': u_guess}, f)


### MPC ###

if MPC_FLAG:
    for k in range(N_init, N_last - 1, -delta):
        data = pickle.load(open(f'{data_dir}{model_name}_{cont_name}_{k}hor_guess.pkl', 'rb'))
        x_guess = data['xg']
        u_guess = data['ug']
        x_init = x_guess[:,0,:]

        # MPC simulation 
        controller.resetHorizon(k)
        conv_idx, collisions_idx, viable_idx = [], [], []
        x_list, u_list, x_viable = [], [], []
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
            sa_flag = False
            for j in range(params.n_steps):
                u[j], sa_flag = controller.step(x_sim[j])
                x_sim[j + 1], _ = model.integrate(x_sim[j], u[j])
                if sa_flag:
                    x_viable += [controller.getLastViableState()]
                    viable_idx.append(i)
                    break

                # Check next state bounds and collision
                if not model.checkStateConstraints(x_sim[j + 1]):
                    collisions_idx.append(i) 
                    break
                if not controller.checkCollision(x_sim[j + 1]):
                    collisions_idx.append(i)
                    break
            # Check convergence
            if np.linalg.norm(model.jointToEE(x_sim[-1]).T - ee_ref) < params.tol_conv:
                conv_idx.append(i)
            x_list.append(x_sim), u_list.append(u)

        unconv_idx = np.setdiff1d(np.arange(params.test_num), 
                                  reduce(np.union1d, (conv_idx, collisions_idx, viable_idx))).tolist()
        
        mpc_data = {'x': np.asarray(x_list), 'u': np.asarray(u_list), 'x_viable': x_viable, 'viable_idx': viable_idx,
                    'collisions_idx': collisions_idx, 'unconv_idx': unconv_idx, 'conv_idx': conv_idx}
        with open(f'{data_dir}{model_name}_{cont_name}_{k}hor_mpc.pkl', 'wb') as f:
            pickle.dump(mpc_data, f)

        print(f'Horizon: {k}, Completed task: {len(conv_idx)}, Collisions: {len(collisions_idx)}, Viable: {len(viable_idx)}')
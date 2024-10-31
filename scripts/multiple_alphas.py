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
# Build the model --> at the moment needed since we modify alpha
params.build = True                     
params.act = args['activation']
model = AdamModel(params, n_dofs=4)
model.ee_ref = ee_ref

ocp_name = args['controller']
if ocp_name in ['stwa', 'htwa', 'receding']:
    pass
else:
    print('No safe abort for the selected controller')
    exit()

cont_name = ocp_name
ocp_class = get_ocp(ocp_name, model, obstacles).__class__
cont_class = get_controller(cont_name, model, obstacles).__class__

data = pickle.load(open(f'{params.DATA_DIR}{model_name}_{cont_name}_guess.pkl', 'rb'))
xg_list = data['xg']
ug_list = data['ug']
x_init = xg_list[:,0,:]
data_dir = params.DATA_DIR + 'ma/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

N = params.N
alphas = [2., 5., 10., 15., 30., 50.]

GUESS_FLAG = True
MPC_FLAG = True


### GUESS ###

if GUESS_FLAG:
    # Generate all the guess for different alpha
    x_tot, u_tot = [], []                   
    for alpha in tqdm(alphas, desc='Testing different alpha'):
        x_alpha, u_alpha = [], []
        params.alpha = alpha
        ocp = ocp_class(model, obstacles)
        opti = ocp.instantiateProblem()

        for i in tqdm(range(params.test_num), desc='Initial condition', leave=False):
            x0 = x_init[i]
            u0 = model.gravity(np.eye(4), x0[:model.nq])[6:]
            
            opti.set_value(ocp.x_init, x0)
            for j in range(N):
                opti.set_initial(ocp.X[j], x0)
                opti.set_initial(ocp.U[j], u0)
            opti.set_initial(ocp.X[-1], x0)

            try:
                sol = opti.solve()
                xg = np.array([sol.value(ocp.X[k]) for k in range(N + 1)])
                ug = np.array([sol.value(ocp.U[k]) for k in range(N)])
                x_alpha.append(xg), u_alpha.append(ug)
            except:
                print(f'Failed at ICs {i} with alpha {alpha}')
                # Keep the guess with the default alpha
                x_alpha.append(xg_list[i]), u_alpha.append(ug_list[i])

        # Store the guess for different alphas
        x_tot.append(x_alpha), u_tot.append(u_alpha)
    
    # Save the guess in different files for each alpha
    for i, alpha in enumerate(alphas):
        x_guess = np.asarray(x_tot[i])
        u_guess = np.asarray(u_tot[i])
        with open(f'{data_dir}{model_name}_receding_{int(alpha)}_guess.pkl', 'wb') as f:
            pickle.dump({'xg': x_guess, 'ug': u_guess}, f)        
    

### MPC ###

if MPC_FLAG:
    for alpha in alphas:
        data = pickle.load(open(f'{data_dir}{model_name}_receding_guess_{alpha}.pkl', 'rb'))
        x_guess = data['xg']
        u_guess = data['ug']
        x_init = x_guess[:,0,:]

        # MPC simulation
        controller = cont_class(model, obstacles)
        controller.setReference(ee_ref)

        x_sim_list, u_list = [], []
        conv_idx, collisions_idx, viable_idx = [], [], []
        for i in tqdm(range(params.test_num), desc='MPC simulations', leave=False):
            x0 = x_init[i]
            x_sim = np.empty((params.n_steps + 1, model.nx)) * np.nan
            u = np.empty((params.n_steps, model.nu)) * np.nan
            x_sim[0] = x0

            controller.setGuess(x_guess[i], u_guess[i])
            try:
                controller.r = N
            except:
                pass
            controller.fails = 0
            j = 0
            for j in range(params.n_steps):
                u[j] = controller.step(x_sim[j])
                x_sim[j + 1], _ = model.integrate(x_sim[j], u[j])
                # Check next state bounds and collision
                if not model.checkStateConstraints(x_sim[j + 1]):
                    if np.isnan(x_sim[j + 1]).any():
                        viable_idx.append(i)
                    else:
                        collisions_idx.append(i)
                    break
                if not controller.checkCollision(x_sim[j + 1]):
                    print(f'Collision at step {j}')
                    collisions_idx.append(i)
                    break
            # Check convergence
            if np.linalg.norm(model.jointToEE(x_sim[-1]).T - ee_ref) < params.tol_conv:
                conv_idx.append(i)

            x_sim_list.append(x_sim), u_list.append(u)

        print(f'Alpha: {alpha}, Completed task: {len(conv_idx)}, 
              Viable, {len(viable_idx)}, Collisions: {len(collisions_idx)}')
        unconv_idx = np.setdiff1d(np.arange(params.test_num), 
                                  reduce(np.union1d, (conv_idx, collisions_idx, viable_idx)))
    
        with open(f'{params.DATA_DIR}{model_name}_{cont_name}_mpc.pkl', 'wb') as f:
            pickle.dump({'x': np.asarray(x_sim_list),
                        'u': np.asarray(u_list),
                        'conv_idx' : np.asarray(conv_idx),
                        'collisions_idx' : np.asarray(collisions_idx),
                        'unconv_idx' : np.asarray(unconv_idx),
                        'viable_idx': np.asarray(viable_idx)}, f)
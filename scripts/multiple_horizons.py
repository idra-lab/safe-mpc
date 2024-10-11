import os
import pickle
import numpy as np
from tqdm import tqdm
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel
from safe_mpc.utils import obstacles, ee_ref, get_controller
from acados_template import AcadosOcpSolver


args = parse_args()
params = Parameters('z1', rti=False)
params.build = True
model = AdamModel(params, n_dofs=4)

cont_name = args['controller']
controller = get_controller(cont_name, model, obstacles)
controller.setReference(ee_ref)

x_init = np.load(params.DATA_DIR + 'ics.npy')
data_dir = params.DATA_DIR + 'mh/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

N_init = params.N
N_last = 20
delta = 5


### GUESS ###

# Generate all the guess for different horizon
x_tot, u_tot = [], []                   # Store all the guess
for i in tqdm(range(len(x_init)), desc='Initial condition'):
    controller.resetHorizon(N_init)
    x0 = x_init[i]
    u0 = model.gravity(np.eye(4), x0[:model.nq])[6:].T
    xg = np.full((controller.N + 1, model.nx), x0)
    ug = np.full((controller.N, model.nu), u0)
    controller.setGuess(xg, ug)
    # Store the guess for different horizons, same IC
    x_hor, u_hor = [], []               
    # for j in range(N_init, N_last - 1, -1):
    for j in tqdm(range(N_init, N_last - 1, -1), desc='Horizon reduction', leave=False):
        controller.resetHorizon(j)
        controller.setGuess(xg[:j + 1], ug[:j])
        status = controller.solve(x0)
        if status == 0 and controller.checkGuess():
            xg = np.copy(controller.x_temp)
            ug = np.copy(controller.u_temp)
        else:
            print(f'IC: {i}, Horizon: {j}, Status: {status}')
            print(model.checkRunningConstraints(controller.x_temp, controller.u_temp), \
              model.checkDynamicsConstraints(controller.x_temp, controller.u_temp), \
                np.all([controller.checkCollision(x) for x in controller.x_temp]))
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

# Build again the OCP with SQP-RTI
params.solver_type = 'SQP_RTI'
params.globalization = 'FIXED_STEP'

del model, controller
model = AdamModel(params, n_dofs=4)
controller = get_controller(cont_name, model, obstacles)
controller.setReference(ee_ref)

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
        controller.fails = 0
        j = 0
        for j in range(params.n_steps):
            u[j] = controller.step(x_sim[j])
            x_sim[j + 1] = model.integrate(x_sim[j], u[j])
            # Check next state bounds and collision
            if not model.checkStateConstraints(x_sim[j + 1]):
                collisions += 1
                break
            if not controller.checkCollision(x_sim[j + 1]):
                collisions += 1
                break
        # Check convergence
        if np.linalg.norm(model.jointToEE(x_sim[-1]).T - ee_ref) < params.conv_tol:
            conv += 1

    print(f'Horizon: {k}, Completed task: {conv}, Collisions: {collisions}')
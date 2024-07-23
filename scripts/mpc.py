import pickle
import numpy as np
import time
from tqdm import tqdm
from urdf_parser_py.urdf import URDF
import adam
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel
from safe_mpc.utils import obstacles, ee_ref
from safe_mpc.controller import NaiveController


args = parse_args()
params = Parameters('z1', True)
params.build = args['build']
model = AdamModel(params, n_dofs=4)

controller = NaiveController(model, obstacles)
controller.setReference(ee_ref)

data = pickle.load(open(params.DATA_DIR + 'guess.pkl', 'rb'))
x_guess = data['x']
u_guess = data['u']
x_init = x_guess[:,0,:]

# MPC simulation 
convergence = 0
collisions = 0
x_sim_list = []
for i in tqdm(range(params.test_num), desc='MPC simulations'):
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
        if not model.checkStateConstraints(x_sim[j + 1]) or not controller.checkCollision(x_sim[j + 1]):
            collisions += 1
            break
        # Check convergence 
        if np.linalg.norm(model.jointToEE(x_sim[j + 1]) - ee_ref) < params.conv_tol:
            convergence += 1
            break
    x_sim_list.append(x_sim)

print('Convergence rate: ', convergence / params.test_num)
print('Number of collisions: ', collisions)

# Save simulation data
with open(params.DATA_DIR + 'mpc_sim.pkl', 'wb') as f:
    pickle.dump({'x': np.asarray(x_sim_list)}, f)
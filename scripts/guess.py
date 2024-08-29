import numpy as np
import pickle
from tqdm import tqdm
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel
from safe_mpc.utils import obstacles, ee_ref, get_controller


args = parse_args()
params = Parameters('z1', rti=False)
params.build = args['build']
model = AdamModel(params, n_dofs=4)

cont_name = args['controller']
controller = get_controller(cont_name, model, obstacles)
controller.setReference(ee_ref)

x_init = np.load(params.DATA_DIR + 'ics.npy')

x_guess, u_guess = [], []
solved = 0
for i in tqdm(range(len(x_init)), desc='Initial condition'):
    x0 = x_init[i]
    u0 = model.gravity(np.eye(4), x0[:model.nq])[6:].T
    if controller.initialize(x0, u0):
        solved += 1
        (xg, ug) = controller.getGuess()
        x_guess.append(xg), u_guess.append(ug)

print(f'Solved {solved} out of {len(x_init)} initial conditions')

x_guess = np.asarray(x_guess)
u_guess = np.asarray(u_guess)

with open(f'{params.DATA_DIR}{cont_name}_guess.pkl', 'wb') as f:
    pickle.dump({'xg': x_guess, 'ug': u_guess}, f)
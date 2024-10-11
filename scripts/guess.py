import numpy as np
import pickle
from tqdm import tqdm
from scipy.stats import qmc
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel, TriplePendulumModel
from safe_mpc.utils import obstacles, ee_ref, get_controller


args = parse_args()
model_name = args['system']
params = Parameters(model_name, rti=False)
params.build = args['build']
if model_name == 'triple_pendulum':
    model = TriplePendulumModel(params)
else:
    model = AdamModel(params, n_dofs=4)

cont_name = args['controller']
controller = get_controller(cont_name, model, obstacles)
controller.setReference(ee_ref)

if model.amodel.name == 'triple_pendulum':
    x_init = np.load(params.DATA_DIR + 'x_init.npy')
else:
    data = pickle.load(open(f'{params.DATA_DIR}initial_guess_tanh.pkl', 'rb'))
    x_ipopt = data['xg']
    u_ipopt = data['ug']
    x_init = x_ipopt[:,0,:]
n = len(x_init)

x_guess, u_guess = [], []
solved = []
for i in tqdm(range(n), desc='Initial condition'):
    if model.amodel.name == 'triple_pendulum':
        x0 = np.zeros((model.nx,))
        x0[:model.nq] = x_init[i]
        if controller.initialize(x0):
            solved.append(i)
        (xg, ug) = controller.getGuess()
        x_guess.append(xg), u_guess.append(ug)

    else:
        # Alternative, use the guess from ipopt
        x0 = x_init[i]
        controller.setGuess(x_ipopt[i], u_ipopt[i])
        status = controller.solve(x0)
        if status == 0:         # controller.checkGuess():
            solved.append(i)
            xg = np.copy(controller.x_temp)
            ug = np.copy(controller.u_temp)
            x_guess.append(xg), u_guess.append(ug)

print(f'Solved {len(solved)} out of {n} initial conditions')
# print(f'Indices: {solved}')
unsolved = n- len(solved)

if unsolved > 0:
    # Generate additional initial conditions
    print(f'\tSearch other {unsolved} ICs')
    i = 0
    fails = 0
    sampler = qmc.Halton(model.nq, scramble=False)
    while i < unsolved:
        q0 = qmc.scale(sampler.random(), model.x_min[:model.nq], model.x_max[:model.nq])[0]
        x0 = np.zeros((model.nx,))
        x0[:model.nq] = q0
        if controller.checkCollision(x0):
            if model.amodel.name == 'triple_pendulum':
                u0 = np.zeros(model.nu)
            else:
                u0 = model.gravity(np.eye(4), q0)[6:].T
            if controller.initialize(x0, u0):
                (xg, ug) = controller.getGuess()
                x_guess.append(xg), u_guess.append(ug)
                i += 1
            else:
                fails += 1
    print(f'\tFound after {fails} fails')

x_guess = np.asarray(x_guess)
u_guess = np.asarray(u_guess)

with open(f'{params.DATA_DIR}{model_name}_{cont_name}_guess.pkl', 'wb') as f:
    pickle.dump({'xg': x_guess, 'ug': u_guess}, f)
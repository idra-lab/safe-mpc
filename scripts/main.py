import os
import pickle
import numpy as np
from tqdm import tqdm
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel
import safe_mpc.controller as controllers


def convergenceCriteria(x, mask=None):
    if mask is None:
        mask = np.ones(model.nx)
    return np.linalg.norm(np.dot(mask, x - model.x_ref)) < params.conv_tol


def init_guess(q0):
    x0 = np.zeros((model.nx,))
    x0[:model.nq] = q0
    # Gravity compensation --> Adam gravity_term_fun
    u0 = model.gravity(np.eye(4), q0)[6:]
    flag = controller.initialize(x0, u0.T)
    return controller.getGuess(), flag


def simulate_mpc(p):
    x0 = np.zeros((model.nx,))
    x0[:model.nq] = x0_vec[p]

    x_sim = np.empty((params.n_steps + 1, model.nx)) * np.nan
    u = np.empty((params.n_steps, model.nu)) * np.nan
    x_sim[0] = x0

    controller.setGuess(x_guess_vec[p], u_guess_vec[p])
    controller.fails = 0
    stats = []
    convergence = 0
    k = 0
    for k in range(params.n_steps):
        u[k] = controller.step(x_sim[k])
        stats.append(controller.getTime())
        x_sim[k + 1] = model.integrate(x_sim[k], u[k])
        # Check if the next state is inside the state bounds
        if not model.checkStateConstraints(x_sim[k + 1]):
            break
        # Check convergence --> norm of diff btw x_sim and x_ref (only for first joint)
        if convergenceCriteria(x_sim[k + 1], np.array([1, 0, 1, 0])):
            convergence = 1
            break
    x_v = controller.getLastViableState()
    return k, convergence, x_sim, stats, x_v


if __name__ == '__main__':
    args = parse_args()
    # Define the available systems and controllers
    available_systems = ['pendulum', 'double_pendulum', 'ur5', 'z1']
    available_controllers = {'naive': 'NaiveController',
                             'st': 'STController',
                             'stwa': 'STWAController',
                             'htwa': 'HTWAController',
                             'receding': 'RecedingController',
                             'abort': 'SafeBackupController'}
    
    # Check if the system and controller selected are available
    try:
        if args['system'] not in available_systems:
            raise NameError
    except NameError:
        print('\nSystem not available! Available: ', available_systems, '\n')
        exit()
        
    try:
        if args['controller'] not in available_controllers.keys():
            raise NameError
    except NameError:
        print('\nController not available! Available: ', available_controllers, '\n')
        exit()

    if args['init_conditions']:
        args['controller'] = 'receding'

    # Define the paramsiguration object, model, simulator and controller
    params = Parameters(args['system'], args['rti'])
    params.build = args['build']
    model = AdamModel(params, n_dofs=args['dofs'])
    controller = getattr(controllers, available_controllers[args['controller']])(model)
    controller.setReference(model.x_ref)

    # Check if data folder exists, if not create it
    if not os.path.exists(params.DATA_DIR):
        os.makedirs(params.DATA_DIR)
    data_name = params.DATA_DIR + args['controller'] + '_'

    print(f'Running {available_controllers[args["controller"]]} with alpha = {params.alpha} ...')
    # If ICs is active, compute the initial conditions for all the controller
    if args['init_conditions']:
        from scipy.stats import qmc

        sampler = qmc.Halton(model.nq, scramble=False)
        l_bounds = model.x_min[:model.nq] + params.state_tol
        u_bounds = model.x_max[:model.nq] - params.state_tol

        # Soft constraints an all the trajectory
        for i in range(1, controller.N):
            controller.ocp_solver.cost_set(i, "zl", params.ws_r * np.ones((1,)))
        controller.ocp_solver.cost_set(controller.N, "zl", params.ws_t * np.ones((1,)))

        h = 0
        failures = 0
        x_init_vec, x_guess_vec, u_guess_vec = [], [], []
        progress_bar = tqdm(total=params.test_num, desc='Init guess processing')
        while h < params.test_num:
            (x_g, u_g), status = init_guess(qmc.scale(sampler.random(), l_bounds, u_bounds))
            if status:
                x_init_vec.append(x_g[0, :model.nq])
                x_guess_vec.append(x_g)
                u_guess_vec.append(u_g)
                h += 1
                progress_bar.update(1)
            else:
                failures += 1

        progress_bar.close()
        # TODO: swap to pickle
        np.save(params.DATA_DIR + f'x_init_{params.alpha}.npy', np.asarray(x_init_vec))
        np.save(params.DATA_DIR + f'x_guess_vec_{params.alpha}.npy', np.asarray(x_guess_vec))
        np.save(params.DATA_DIR + f'u_guess_vec_{params.alpha}.npy', np.asarray(u_guess_vec))
        print(f'Found {params.test_num} initial conditions after {failures} failures.')

    elif args['guess']:
        x_init_vec = np.load(params.DATA_DIR + f'x_init_{params.alpha}.npy')
        x_guess_vec, u_guess_vec, successes = [], [], []

        if args['controller'] in ['naive', 'st']:
            for x_init in x_init_vec:
                (x_g, u_g), status = init_guess(x_init)
                x_guess_vec.append(x_g)
                u_guess_vec.append(u_g)
                successes.append(status)
        else:
            x_feasible = np.load(params.DATA_DIR + f'x_guess_vec_{params.alpha}.npy')
            u_feasible = np.load(params.DATA_DIR + f'u_guess_vec_{params.alpha}.npy')
            # Try to refine the guess with respect to the controller used
            for i in range(params.test_num):
                controller.setGuess(x_feasible[i], u_feasible[i])
                x_init = np.zeros((model.nx,))
                x_init[:model.nq] = x_init_vec[i]
                status = controller.solve(x_init)
                if (status == 0 or status == 2) and controller.checkGuess():
                    # Refinement successful
                    x_g, u_g = controller.getGuess()
                    x_guess_vec.append(x_g)
                    u_guess_vec.append(u_g)
                    successes.append(1)
                else:
                    # Refinement failed, use the feasible guess
                    x_guess_vec.append(x_feasible[i])
                    u_guess_vec.append(u_feasible[i])
        np.save(data_name + 'x_guess.npy', np.asarray(x_guess_vec))
        np.save(data_name + 'u_guess.npy', np.asarray(u_guess_vec))
        print('Init guess success: %d over %d' % (sum(successes), params.test_num))

    elif args['rti']:
        x0_vec = np.load(params.DATA_DIR + f'x_init_{params.alpha}.npy')
        x_guess_vec = np.load(data_name + 'x_guess.npy')
        u_guess_vec = np.load(data_name + 'u_guess.npy')
        res = []
        for i in range(params.test_num):
            res.append(simulate_mpc(i))
        steps, conv_vec, x_sim_vec, t_stats, x_viable = zip(*res)
        steps = np.array(steps)
        conv_vec = np.array(conv_vec)
        idx = np.where(conv_vec == 1)[0]
        idx_abort = np.where(conv_vec == 0)[0]
        print('Total convergence: %d over %d' % (np.sum(conv_vec), params.test_num))

        print('99% quantile computation time:')
        times = np.array([t for arr in t_stats for t in arr])
        for field, t in zip(controller.time_fields, np.quantile(times, 0.99, axis=0)):
            print(f"{field:<20} -> {t}")

        # Save last viable states (useful only for terminal/receding controllers)
        np.save(data_name + 'x_viable.npy', np.asarray(x_viable)[idx_abort])

        # Save all the results
        with open(data_name + 'results.pkl', 'wb') as f:
            pickle.dump({'x_sim': np.asarray(x_sim_vec),
                         'times': times,
                         'steps': steps,
                         'idx_abort': idx_abort,
                         'x_viable': np.asarray(x_viable)}, f)

    elif args['controller'] == 'abort' and args['abort'] in ['stwa', 'htwa', 'receding']:
        # Increase time horizon
        x_viable = np.load(params.DATA_DIR + args['abort'] + '_' + 'x_viable.npy')
        n_a = np.shape(x_viable)[0]
        rep = args['repetition']
        t_rep = np.empty((n_a, rep)) * np.nan
        for i in range(n_a):
            # Repeat each test rep times
            for j in range(rep):
                controller.setGuess(np.full((controller.N + 1, model.nx), x_viable[i]),
                                    np.zeros((controller.N, model.nu)))
                status = controller.solve(x_viable[i])
                if status == 0:
                    t_rep[i, j] = controller.ocp_solver.get_stats('time_tot')[0]
        # Compute the minimum time for each initial condition
        t_min = np.min(t_rep, axis=1)
        # Remove the nan values (i.e. the initial conditions for which the controller failed)
        t_min = t_min[~np.isnan(t_min)]
        print('Controller: %s\nAbort: %d over %d\nQuantile (99) time: %.3f'
              % (args['abort'], len(t_min), n_a, np.quantile(t_min, 0.99)))
    else:
        pass

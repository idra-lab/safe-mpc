import os
import shutil
import numpy as np
import safe_mpc.model as models
import safe_mpc.controller as controllers
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import SimDynamics
from safe_mpc.gravity_compensation import GravityCompensation


def convergenceCriteria(x, mask=None):
    if mask is None:
        mask = np.ones(model.nx)
    return np.linalg.norm(np.dot(mask, x - x_ref)) < conf.conv_tol


def init_guess(n):
    x_sol_vec, u_sol_vec = [], []
    success_list = []
    for k in range(n):
        x0 = np.zeros((model.nx,))
        x0[:model.nq] = x0_vec[k]
        u0 = gc.solve(x0)

        flag = controller.initialize(x0, u0)
        xg, ug = controller.getGuess()
        x_sol_vec.append(xg)
        u_sol_vec.append(ug)
        success_list.append(flag)
            
    return success_list, x_sol_vec, u_sol_vec


def simulate_mpc(p):
    x0 = np.zeros((model.nx,))
    x0[:model.nq] = x0_vec[p]

    x_sim = np.empty((conf.n_steps + 1, model.nx)) * np.nan
    u = np.empty((conf.n_steps, model.nu)) * np.nan
    x_sim[0] = x0

    controller.setGuess(x_guess_vec[p], u_guess_vec[p])
    controller.fails = 0
    stats = []
    convergence = 0
    k = 0
    for k in range(conf.n_steps):
        u[k] = controller.step(x_sim[k])
        # stats.append(controller.getTime())
        x_sim[k + 1] = simulator.simulate(x_sim[k], u[k])
        # Check if the next state is inside the state bounds
        if not model.checkStateConstraints(x_sim[k + 1]):
            break
        # Check convergence --> norm of diff btw x_sim and x_ref (only for first joint)
        if convergenceCriteria(x_sim[k + 1], np.array([1, 0, 0, 1, 0, 0])):
            convergence = 1
            break
    x_v = controller.getLastViableState()
    return k, convergence, x_sim, stats, x_v


if __name__ == '__main__':
    args = parse_args()
    # Define the available systems and controllers
    available_systems = {'double_pendulum': 'DoublePendulumModel',
                         'triple_pendulum': 'TriplePendulumModel'}
    available_controllers = {'naive': 'NaiveController',
                             'st': 'STController',
                             'stwa': 'STWAController',
                             'htwa': 'HTWAController',
                             'receding': 'RecedingController',
                             'abort': 'SafeBackupController'}

    if args['init_conditions']:
        args['controller'] = 'receding'
    # Check if the system and controller selected are available
    if args['system'] not in available_systems.keys():
        raise ValueError('Unknown system. Available: ' + str(available_systems))
    if args['controller'] not in available_controllers.keys():
        raise ValueError('Unknown controller. Available: ' + str(available_controllers))

    # Define the configuration object, model, simulator and controller
    conf = Parameters(args['system'], args['controller'], args['rti'])
    # Set the safety margin
    conf.alpha = args['alpha']
    model = getattr(models, available_systems[args['system']])(conf)
    gc = GravityCompensation(conf, model)
    simulator = SimDynamics(model)
    controller = getattr(controllers, available_controllers[args['controller']])(simulator)

    x_ref = np.array([conf.q_max - 0.05, np.pi, np.pi, 0, 0, 0])
    controller.setReference(x_ref)

    # Check if data folder exists, if not create it
    if not os.path.exists(conf.DATA_DIR):
        os.makedirs(conf.DATA_DIR)
    data_name = conf.DATA_DIR + args['controller'] + '_'

    # If ICs is active, compute the initial conditions for all the controller
    if args['init_conditions']:
        from scipy.stats import qmc
        test_num = 500                      # Higher than the one reported in the configuration file
        sampler = qmc.Halton(d=controller.ocp.dims.nu, scramble=False)
        sample = sampler.random(n=test_num)
        l_bounds = model.x_min[:model.nq] + conf.state_tol
        u_bounds = model.x_max[:model.nq] - conf.state_tol
        x0_vec = qmc.scale(sample, l_bounds, u_bounds)

        # Soft constraints an all the trajectory
        for i in range(controller.N):
            controller.ocp_solver.cost_set(i, "zl", conf.ws_r * np.ones((1,)))
        controller.ocp_solver.cost_set(controller.N, "zl", conf.ws_t * np.ones((1,)))

        overall_success, x_guess_vec, u_guess_vec = init_guess(test_num)
        # Save the results -> only the first conf.test_num
        success_arr = np.asarray(overall_success)
        idx = np.where(success_arr == 1)[0][:conf.test_num]

        np.save(conf.DATA_DIR + 'x_init.npy', np.asarray(x0_vec)[idx])
        np.save(conf.DATA_DIR + 'x_guess_vec.npy', np.asarray(x_guess_vec)[idx])
        np.save(conf.DATA_DIR + 'u_guess_vec.npy', np.asarray(u_guess_vec)[idx])
        print('Init guess success: %d over %d' % (np.sum(success_arr), test_num))

    elif args['guess']:
        if args['controller'] in ['naive', 'st']:
            x0_vec = np.load(conf.DATA_DIR + 'x_init.npy')
            overall_success, x_guess_vec, u_guess_vec = init_guess(conf.test_num)
            np.save(data_name + 'x_guess.npy', np.asarray(x_guess_vec))
            np.save(data_name + 'u_guess.npy', np.asarray(u_guess_vec))
            print('Init guess success: %d over %d' % (sum(overall_success), conf.test_num))
        else:
            # Just copy the initial guess in the corresponding controller folder
            shutil.copy(conf.DATA_DIR + 'x_guess_vec.npy', data_name + 'x_guess.npy')
            shutil.copy(conf.DATA_DIR + 'u_guess_vec.npy', data_name + 'u_guess.npy')

    elif args['rti']:
        x0_vec = np.load(conf.DATA_DIR + 'x_init.npy')
        x_guess_vec = np.load(data_name + 'x_guess.npy')
        u_guess_vec = np.load(data_name + 'u_guess.npy')
        res = []
        for i in range(conf.test_num):
            res.append(simulate_mpc(i))
        steps, conv_vec, x_sim_vec, t_stats, x_viable = zip(*res)
        steps = np.array(steps)
        conv_vec = np.array(conv_vec)
        idx = np.where(conv_vec == 1)[0]
        idx_abort = np.where(conv_vec == 0)[0]
        print('Total convergence: %d over %d' % (np.sum(conv_vec), conf.test_num))

        print('99% quantile computation time:')
        times = np.array([t for arr in t_stats for t in arr])
        for field, t in zip(controller.time_fields, np.quantile(times, 0.99, axis=0)):
            print(f"{field:<20} -> {t[0]}")

        # Save last viable states (useful only for terminal/receding controllers)
        np.save(data_name + 'x_viable.npy', np.asarray(x_viable)[idx_abort])

    elif args['controller'] == 'abort' and args['abort'] in ['stwa', 'htwa', 'receding']:
        # Increase time horizon
        x_viable = np.load(conf.DATA_DIR + args['abort'] + '_' + 'x_viable.npy')
        n_a = np.shape(x_viable)[0]
        rep = args['repetition']
        t_rep = np.empty((n_a, rep)) * np.nan
        for i in range(n_a):
            # Repeat each test rep times
            for j in range(rep):
                controller.setGuess(np.full((controller.N + 1, model.nx), x_viable[i]),
                                    np.zeros((controller.N, model.nu)))
                status = controller.solve(x_viable[i])
                print(status)
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

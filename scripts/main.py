import os
import argparse
import numpy as np
from safe_mpc.parser import Parameters
import safe_mpc.model as models
from safe_mpc.abstract import SimDynamics
import safe_mpc.controller as controllers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--system', type=str, default='triple_pendulum',
                        help='Systems to test. Available: double_pendulum, triple_pendulum')
    parser.add_argument('-c', '--controller', type=str, default='naive',
                        help='Controllers to test. Available: naive, st, stwa, htwa, receding')
    parser.add_argument('-i', '--init-conditions', action='store_true',
                        help='Find the initial conditions for testing all the controller')
    parser.add_argument('-g', '--guess', action='store_true',
                        help='Compute the initial guess of a given controller')
    parser.add_argument('--rti', action='store_true',
                        help='Use SQP-RTI for the MPC solver')
    parser.add_argument('--plot', action='store_true',
                        help='Plot the results')
    return vars(parser.parse_args())


def init_guess(n):
    x_sol_vec, u_sol_vec = [], []
    total_success = 0
    for k in range(n):
        x0 = np.zeros((model.nx,))
        x0[:model.nq] = x0_vec[k]

        success = controller.initialize(x0)
        if success == 1:
            xg, ug = controller.getGuess()
            x_sol_vec.append(xg)
            u_sol_vec.append(ug)
            total_success += 1
            
    return total_success, x_sol_vec, u_sol_vec


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
    j = 0
    for j in range(conf.n_steps):
        u[j] = controller.step(x_sim[j])
        stats.append(controller.getTime())
        x_sim[j + 1] = simulator.simulate(x_sim[j], u[j])
        # Check if the next state is inside the state bounds
        if not model.checkStateConstraints(x_sim[j + 1]):
            break
        # Check convergence --> norm of diff btw x_sim and x_ref (only for first joint)
        if np.linalg.norm([x_sim[j + 1, 0] - x_ref[0], x_sim[j + 1, 3]]) < 1e-3:
            # TODO: remove hard-coded tolerance
            # TODO: use a general convergence criteria
            convergence = 1
            break
    return j, convergence, x_sim, stats


if __name__ == '__main__':
    args = parse_args()
    # Define the available systems and controllers
    available_systems = {'double_pendulum': 'DoublePendulumModel',
                         'triple_pendulum': 'TriplePendulumModel'}
    available_controllers = {'naive': 'NaiveController',
                             'st': 'STController',
                             'stwa': 'STWAController',
                             'htwa': 'HTWAController',
                             'receding': 'RecedingController'}
    if args['init_conditions']:
        args['controller'] = 'receding'
    # Check if the system and controller selected are available
    if args['system'] not in available_systems.keys():
        raise ValueError('Unknown system. Available: ' + str(available_systems))
    if args['controller'] not in available_controllers.keys():
        raise ValueError('Unknown controller. Available: ' + str(available_controllers))
    # Define the configuration object, model, simulator and controller
    conf = Parameters(args['system'], args['controller'], args['rti'])
    model = getattr(models, available_systems[args['system']])(conf)
    simulator = SimDynamics(model)
    controller = getattr(controllers, available_controllers[args['controller']])(simulator)

    x_ref = np.array([conf.q_max - 0.05, np.pi, np.pi, 0, 0, 0])
    controller.setReference(x_ref)

    # Check if data folder exists, if not create it
    if not os.path.exists(conf.DATA_DIR):
        os.makedirs(conf.DATA_DIR)
    ics_dir = os.path.join(conf.DATA_DIR, args['controller'])
    if not os.path.exists(ics_dir):
        os.makedirs(ics_dir)

    # If ICs is active, compute the initial conditions for all the controller
    if args['init_conditions']:
        from scipy.stats import qmc
        test_num = 500                      # Higher than the one reported in the configuration file
        sampler = qmc.Halton(d=controller.ocp.dims.nu, scramble=False)
        sample = sampler.random(n=test_num)
        l_bounds = model.x_min[:model.nq]
        u_bounds = model.x_max[:model.nq]
        x0_vec = qmc.scale(sample, l_bounds, u_bounds)

        # Soft constraints an all the trajectory
        for i in range(conf.N):
            controller.ocp_solver.cost_set(i, "zl", conf.ws_r * np.ones((1,)))
        controller.ocp_solver.cost_set(conf.N, "zl", conf.ws_t * np.ones((1,)))

        overall_success, x_guess_vec, u_guess_vec = init_guess(test_num)
        # Save the results
        np.save(conf.DATA_DIR + '/x_init.npy', np.asarray(x0_vec)[:conf.test_num])
        np.save(ics_dir + '/x_guess_vec.npy', np.asarray(x_guess_vec)[:conf.test_num])
        np.save(ics_dir + '/u_guess_vec.npy', np.asarray(u_guess_vec)[:conf.test_num])
        print('Init guess success: ' + str(overall_success) + ' over ' + str(test_num))
    elif args['guess']:
        x0_vec = np.load(conf.DATA_DIR + '/x_init.npy')
        overall_success, x_guess_vec, u_guess_vec = init_guess(conf.test_num)
        np.save(ics_dir + '/x_guess_vec.npy', np.asarray(x_guess_vec))
        np.save(ics_dir + '/u_guess_vec.npy', np.asarray(u_guess_vec))
        print('Init guess success: ' + str(overall_success) + ' over ' + str(conf.test_num))
    elif args['rti']:
        x0_vec = np.load(conf.DATA_DIR + '/x_init.npy')
        x_guess_vec = np.load(ics_dir + '/x_guess_vec.npy')
        u_guess_vec = np.load(ics_dir + '/u_guess_vec.npy')
        res = []
        for i in range(conf.test_num):
            res.append(simulate_mpc(i))
        steps, conv_vec, x_sim_vec, t_stats = zip(*res)
        steps = np.array(steps)
        conv_vec = np.array(conv_vec)
        idx = np.where(conv_vec == 1)[0]
        times = np.array([t for arr in t_stats for t in arr])
        print('Total convergence: ' + str(np.sum(conv_vec)) + ' over ' + str(conf.test_num))
    else:
        pass

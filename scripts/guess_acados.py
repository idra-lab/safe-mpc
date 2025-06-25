import time
import pickle
import numpy as np
from tqdm import tqdm
from scipy.stats import qmc
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel
from safe_mpc.utils import  get_controller , get_ocp_acados, randomize_model
from safe_mpc.robot_visualizer import RobotVisualizer
from safe_mpc.ocp import InverseKinematicsOCP
import copy
from safe_mpc.cost_definition import *

args = parse_args()
model_name = args['system']
params = Parameters(args,model_name, rti=False)
params.q_margin = args['joint_bounds_margin']
params.collision_margin = args['collision_margin']
params.build = args['build']
params.solver_type = 'SQP'
params.act = args['activation']
params.alpha = args['alpha']
params.N=args['horizon']

params.noise_mass = args['noise']
params.noise_inertia = args['noise']
params.noise_cm = args['noise']

model = AdamModel(params)

# cost = TrackingMovingCircleNLS(model,params.Q_weight,params.R_weight)
#cost = Tracking8NLS(model,params.Q_weight,params.R_weight)
cost = ReachTargetEXT(model,params.Q_weight,params.R_weight)
cost_naive_zerovel = ReachTargetNLS(model,params.Q_weight,params.R_weight)
ocp_name = args['controller']
params.cont_name = args['controller']

build_controllers = args['build']

ocp_with_net, controllers_list = get_ocp_acados(ocp_name, model)
ocp_with_net.set_cost(cost)
ocp_with_net.build_controller(build = build_controllers)
ocp_with_net.resetHorizon(args['horizon'])

params_naive = Parameters(args,model_name, rti=False)
params_naive.build = args['build']
params_naive.solver_type = 'SQP'
params_naive.N=args['horizon']
params_naive.q_margin = args['joint_bounds_margin']
params_naive.collision_margin = args['collision_margin']
model_naive = AdamModel(params_naive)

params_zerovel = Parameters(args,model_name, rti=False)
params_zerovel.build = args['build']
params_zerovel.solver_type = 'SQP'
params_zerovel.N=args['horizon']
params_zerovel.q_margin = args['joint_bounds_margin']
params_zerovel.collision_margin = args['collision_margin']
model_zerovel = AdamModel(Parameters(args,model_name, rti=False))

ocp_naive,_ = get_ocp_acados('naive',model_naive)
ocp_naive.set_cost(cost_naive_zerovel)
ocp_naive.build_controller(build = build_controllers)
ocp_naive.resetHorizon(args['horizon'])

ocp_zerovel,_ = get_ocp_acados('zerovel',model_zerovel)
ocp_zerovel.set_cost(cost_naive_zerovel)
ocp_zerovel.build_controller(build = build_controllers)
ocp_zerovel.resetHorizon(args['horizon'])





num_ics = params.test_num
succ, fails, skip_ics = 0, 0, 0
sampler = qmc.Halton(model.nq, scramble=False)

x_guess_net, u_guess_net = [], []
x_guess_naive, u_guess_naive = [], []
x_guess_zerovel, u_guess_zerovel = [], []

# rviz = RobotVisualizer(params, params.nq)
# if params.obs_flag:
#     rviz.addObstacles(params.obstacles)
# rviz.init_capsule(params.robot_capsules+params.obst_capsules)
# rviz.init_spheres(params.spheres_robot)

print(f'Use network: {ocp_with_net.model.params.use_net}')

TEST_NOISE = True

progress_bar = tqdm(total=num_ics, desc=f'Generating initial conditions, alpha {ocp_with_net.model.params.alpha}')
start_time = time.time()
if not(ocp_with_net.model.params.track_traj):
    while succ < num_ics:

        q0 = qmc.scale(sampler.random(), model.x_min[:model.nq], model.x_max[:model.nq])[0]
        if TEST_NOISE:
            q0 = np.array([-0.6,2.513,-2.3,0.272])
            #q0 = np.array([0.,0.,-0.226,0.])

        x0 = np.zeros((model.nx,))
        x0[:model.nq] = q0

        #rviz.displayWithEESphere(x0[:params.nq],params.robot_capsules+params.obst_capsules,params.spheres_robot)
        if ocp_with_net.model.checkCollision(x0):
            print(f'accepted:{x0}')
            u0_g = np.array([np.zeros((model.nu,))]*ocp_with_net.N)
            x0_g = np.array([x0]*(args['horizon']+1))
            ocp_with_net.setGuess(x0_g,u0_g)

            status = ocp_with_net.solve(x0)
            print(f'Status: {status} , Check guess: {ocp_with_net.checkGuess()}')
            print(f'Final velocity: {np.linalg.norm(ocp_with_net.x_temp[args["horizon"],model.nq:])}')
            if (status == 0 or status == 2) and ocp_with_net.checkGuess():
                print(f'Solver status {status}, x0 {x0}')
                xg_net = copy.copy(ocp_with_net.x_temp)
                ug_net = copy.copy(ocp_with_net.u_temp)
                x_guess_net.append(xg_net), u_guess_net.append(ug_net)
                succ += 1
                progress_bar.update(1)
                print('SUCCESS')
                if 'analytic' in args['controller']:
                    print(f'Check analytic constraint {ocp_with_net.model.checkAnalyticConstraints(ocp_with_net.x_temp[-1],ocp_with_net.obstacles)}')
                if args['controller'] == 'htwa':
                    print(f'Check network constraint {ocp_with_net.model.checkSafeConstraints(ocp_with_net.x_temp[-1])}')
                    print(f'Value: {ocp_with_net.model.nn_func(ocp_with_net.x_temp[-1], ocp_with_net.params.alpha)}')

                ocp_naive.setGuess(x0_g,u0_g)
                ocp_zerovel.setGuess(x0_g,u0_g)
                status_naive = ocp_naive.solve(x0)
                if ((status_naive == 0 or status_naive == 2 ) and ocp_naive.checkGuess()):
                    x_guess_naive.append(copy.copy(ocp_naive.x_temp))
                    u_guess_naive.append(copy.copy(ocp_naive.u_temp))
                else:
                    x_guess_naive.append(copy.copy(ocp_with_net.x_temp))
                    u_guess_naive.append(copy.copy(ocp_with_net.u_temp))

                status_zero_vel = ocp_zerovel.solve(x0)
                if ((status_zero_vel==0 or status_zero_vel==2) and ocp_zerovel.checkGuess()):
                    x_guess_zerovel.append(copy.copy(ocp_zerovel.x_temp))
                    u_guess_zerovel.append(copy.copy(ocp_zerovel.u_temp))
                else:
                    x_guess_zerovel.append(copy.copy(ocp_with_net.x_temp))
                    u_guess_zerovel.append(copy.copy(ocp_with_net.u_temp))

                if TEST_NOISE:
                    for ll in range(params.test_num-1):
                        x_guess_net.append(xg_net)
                        u_guess_net.append(ug_net)
                        x_guess_zerovel.append(copy.copy(ocp_zerovel.x_temp))
                        u_guess_zerovel.append(copy.copy(ocp_zerovel.u_temp))
                        x_guess_naive.append(copy.copy(ocp_naive.x_temp))
                        u_guess_naive.append(copy.copy(ocp_naive.u_temp)) 
                    break   

            else:
                fails += 1
                print(f'FAILED !!!')
        else:
            print(f'Skipped:{x0}')
            time.sleep(5)
            skip_ics += 1
else:
    rviz.addTraj(ocp_with_net.cost.traj)
    rviz.vizTraj(ocp_with_net.cost.traj)

    count=0
    while succ < num_ics:
        randomize_model(params.robot_urdf, noise_mass = params.noise_mass, noise_inertia = params.noise_inertia, noise_cm_position = params.noise_cm)
        ocp_with_net.model.update_randomized_dynamics()
        ocp_naive.model.update_randomized_dynamics()
        ocp_zerovel.model.update_randomized_dynamics()


        InvKynSolver = InverseKinematicsOCP(model,ocp_with_net.cost.traj[:,0])
        solver_inv = InvKynSolver.instantiateProblem()
        rviz.displayWithEESphere(np.zeros(ocp_with_net.model.nq),params.robot_capsules+params.obst_capsules,params.spheres_robot)
        sol = solver_inv.solve()
        x0 = sol.value(InvKynSolver.X[0])

        rviz.displayWithEESphere(x0[:ocp_with_net.model.nq],params.robot_capsules+params.obst_capsules,params.spheres_robot)

        u0_g = np.array([np.zeros((model.nu,))]*args['horizon'])
        x0_g = np.array([x0]*(args['horizon']+1))

        ocp_with_net.resetHorizon(args['horizon'])
        ocp_with_net.setGuess(x0_g,u0_g)

        count += 1
        print(f'Configuration {count}/{num_ics}')
        status = ocp_with_net.solve(x0)
        print(f'status : {status}')
        if (status == 0 or status==2) and ocp_with_net.checkGuess():
            print(f'Solver trajectory tracking initialization status {status}, x0 {x0}')
            xg_net = copy.copy(ocp_with_net.x_temp)
            ug_net = copy.copy(ocp_with_net.u_temp)
            x_guess_net.append(xg_net), u_guess_net.append(ug_net)
            succ += 1
            print('SUCCESS')

            ocp_naive.resetHorizon(args['horizon'])
            ocp_naive.setGuess(x0_g,u0_g)

            ocp_zerovel.resetHorizon(args['horizon'])
            ocp_zerovel.setGuess(x0_g,u0_g)
            status_naive = ocp_naive.solve(x0)
            if ((status_naive ==0 or status_naive == 2 ) and ocp_naive.checkGuess()):
                x_guess_naive.append(copy.copy(ocp_naive.x_temp))
                u_guess_naive.append(copy.copy(ocp_naive.u_temp))
            else:
                x_guess_naive.append(copy.copy(ocp_with_net.x_temp))
                u_guess_naive.append(copy.copy(ocp_with_net.u_temp))

            status_zero_vel = ocp_zerovel.solve(x0)
            if ((status_zero_vel==0 or status_zero_vel==2) and ocp_zerovel.checkGuess()):
                x_guess_zerovel.append(copy.copy(ocp_zerovel.x_temp))
                u_guess_zerovel.append(copy.copy(ocp_zerovel.u_temp))
            else:
                x_guess_zerovel.append(copy.copy(ocp_with_net.x_temp))
                u_guess_zerovel.append(copy.copy(ocp_with_net.u_temp))
        else:
            print('FAILED')

progress_bar.close()

print(f'Number of failed initializations: {fails}')
print(f'Number of skipped initial conditions: {skip_ics}')

traj__track = 'traj_track' if ocp_with_net.model.params.track_traj else ""

with open(f'{params.DATA_DIR}{model_name}_naive_{args["horizon"]}hor_{int(params.alpha)}sm_use_net{ocp_naive.model.params.use_net}_{traj__track}_q_collision_margins_{params_naive.q_margin}_{params_naive.collision_margin}_guess.pkl', 'wb') as f:
        pickle.dump({'xg': np.asarray(x_guess_naive), 'ug': np.asarray(u_guess_naive)}, f)
with open(f'{params.DATA_DIR}{model_name}_zerovel_{args["horizon"]}hor_{int(params.alpha)}sm_use_net{ocp_zerovel.model.params.use_net}_{traj__track}_q_collision_margins_{params_zerovel.q_margin}_{params_zerovel.collision_margin}_guess.pkl', 'wb') as f:
        pickle.dump({'xg': np.asarray(x_guess_zerovel), 'ug': np.asarray(u_guess_zerovel)}, f)

if (args['controller']!= 'naive' and args['controller']!= 'zerovel'):
    for cont in controllers_list:
        if cont in ['st','stwa','htwa','receding','real_receding','parallel','parallel2']:
            with open(f'{params.DATA_DIR}{model_name}_{cont}_{args["horizon"]}hor_{int(params.alpha)}sm_use_net{ocp_with_net.model.params.use_net}_{traj__track}_q_collision_margins_{params.q_margin}_{params.collision_margin}_guess.pkl', 'wb') as f:
                pickle.dump({'xg': np.asarray(x_guess_net), 'ug': np.asarray(u_guess_net)}, f)


elapsed_time = time.time() - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)
print(f'Elapsed time: {hours}:{minutes:2d}:{seconds:2d}')

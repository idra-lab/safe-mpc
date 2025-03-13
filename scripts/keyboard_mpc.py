import time
import pickle
import numpy as np
from pynput import keyboard
import multiprocessing as mp    
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel
from safe_mpc.utils import get_ocp, get_controller, ee_ref, R_ref, obstacles, \
                          RobotVisualizer, capsules, capsule_pairs
from safe_mpc.controller import SafeBackupController


def save_log(*args):
    logs = {
        'x': np.array(args[0]),
        'u': np.array(args[1]),
        'r': np.array(args[2]),
        'bounds': np.array(args[3]),
        'coll': np.array(args[4]),
        'solver': np.array(args[5]),
        'nn_r': np.array(args[6]),
        'nn_last': np.array(args[7]),
        'solver_time': np.array(args[8]),
        'tot_time': np.array(args[9])
    }
    with open('logs.pkl', 'wb') as f:
        pickle.dump(logs, f)


def run_mpc(queue, xg, ug):

    keys = []
    def on_press(key):
        try:
            k = key.char
        except:
            k = key.name
        if(k in ['left', 'right', 'up', 'down', 'page_up', 'page_down', 'q']):
            keys.append(k)
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # LOGSSSS
    LOGS = 1. if controller.ocp_name == "receding" else 0.
    x_log, u_log, r_log = [], [], []
    bounds_log, coll_log, solver_log = [], [], []
    nn_r_log, nn_last_log = [], [] 
    solver_time, tot_time = [], []

    x = x0
    controller.setGuess(xg, ug)
    controller.resetHorizon(params.N)
    ia, sa_flag = 0, False

    if LOGS:
        x_log.append(x0), r_log.append(controller.N)
        nn_last = float(model.nn_func(xg[-1], params.alpha))
        nn_r_log.append(nn_last), nn_last_log.append(nn_last)

    step_size = 0.02
    omega = 6.28*1.5
    amp = 0.07
    t = 0.0
    sin_ref = np.copy(ee_ref)
    use_sinusoid = 0

    i = 0
    while 1:

        start_time = time.time()

        if(use_sinusoid):
            sin_ref[0] = ee_ref[0]
            sin_ref[1] = ee_ref[1] + amp*np.sin(omega*t)
            sin_ref[2] = ee_ref[2] + amp*np.cos(omega*t)
            controller.setReference(sin_ref)
            t += params.dt

        try:
            k = keys.pop(0)
            if(k=="up"):      
                ee_ref[0] -= step_size
            elif(k=="down"):   
                ee_ref[0] += step_size
            elif(k=="right"):      
                ee_ref[1] += step_size
            elif(k=="left"):   
                ee_ref[1] -= step_size
            elif(k=="page_up"):      
                ee_ref[2] += step_size
            elif(k=="page_down"):   
                ee_ref[2] -= step_size
            elif(k=="q"):
                print("QUITTING...")
                save_log(x_log, u_log, r_log, bounds_log, coll_log, solver_log, 
                         nn_r_log, nn_last_log, solver_time, tot_time)
                break
            print("\nTarget", ee_ref)
            if(not use_sinusoid):
                controller.setReference(ee_ref)
        except:
            pass

        if sa_flag: #and ia < safe_ocp.N:
            # u = u_abort[ia]
            # ia += 1
            if ia < safe_ocp.N:
                u = u_abort[ia]
                ia += 1
            else:
                u = np.zeros(model.nu)
                u = kp * (x_abort[-1, :nq] - x[:nq]) - kd * x[nq:]
                
        else:
            u, sa_flag = controller.step(x)
            solver_time.append(controller.ocp_solver.get_stats("time_tot"))

            if sa_flag:
                print(f'  ABORT at step {i}, u = {u}')
                x_viable = controller.getLastViableState()
                xg = np.full((safe_ocp.N + 1, model.nx), x_viable)
                ug = np.zeros((safe_ocp.N, model.nu))
                safe_ocp.setGuess(xg, ug) 
                status = safe_ocp.solve(x_viable)
                if status != 0:
                    print('  SAFE ABORT FAILED')
                    print('  Current controller fails:', controller.fails)
                    save_log(x_log, u_log, r_log, bounds_log, coll_log, solver_log, 
                             nn_r_log, nn_last_log, solver_time, tot_time)
                    break
                ia = 0 
                x_abort, u_abort = safe_ocp.x_temp, safe_ocp.u_temp
        
        # Integrate
        x_next, _ = model.integrate(x, u)

        # Check next state bounds and collision
        if not model.checkStateConstraints(x_next):   
            print('  FAIL BOUNDS')
            print(f'\tState {i + 1} violation: {np.min(np.vstack((model.x_max - x_next, x_next - model.x_min)), axis=0)}')
            print(f'\tCurrent controller fails: {controller.fails}')
            save_log(x_log, u_log, r_log, bounds_log, coll_log, solver_log, 
                     nn_r_log, nn_last_log, solver_time, tot_time)
            break
        if not controller.checkCollision(x_next):
            print('  FAIL COLLISION')
            print(x_next)
            print(i)
            save_log(x_log, u_log, r_log, bounds_log, coll_log, solver_log, 
                     nn_r_log, nn_last_log, solver_time, tot_time)
            break
        
        x = x_next

        end_time = time.time()
        tot_time.append(end_time - start_time)

        delta = params.dt - (end_time - start_time)
        time.sleep(delta if delta > 0 else 0)
        ref = sin_ref if use_sinusoid else ee_ref
        if not queue.full():
            queue.put((x[:nq], ref)) 
        i += 1

        # if i % 100 == 0:
        #     x_ee = np.empty(controller.N + 1)
        #     for k in range(controller.N + 1):
        #         x_ee[k] = controller.ocp_solver.get(k, "p")[0]
        #     # print(f'Step {i}, current parameters in N: {controller.ocp_solver.get(controller.N, "p")}')
        #     print(f'Step {i}, current ee ref in x:\n{x_ee}')

        #### LOG THE SHIT #### 
        if LOGS:
            x_log.append(x), u_log.append(u), r_log.append(controller.r)
            bounds_log.append(1. if model.checkStateConstraints(controller.x_temp) else 0.)
            coll_log.append(1. if np.all([controller.checkCollision(x) for x in controller.x_temp]) else 0.)
            solver_log.append(controller.last_status)
            nn_r_log.append(float(model.nn_func(controller.x_temp[controller.r], params.alpha)))
            nn_last_log.append(float(model.nn_func(controller.x_temp[-1], params.alpha)))


def run_visualizer(queue):
    rviz = RobotVisualizer(params, nq)
    rviz.display(x0[:nq])
    if params.obs_flag:
        rviz.addObstacles(obstacles)
        for capsule in controller.capsules:
            rviz.init_capsule(capsule)
    while 1:
        if not queue.empty():
            x, ref = queue.get()
            rviz.displayWithEESphere(x, controller.capsules)
            rviz.setTarget(ref)
        time.sleep(0.01)

if __name__ == "__main__":

    args = parse_args()
    model_name = args['system']
    params = Parameters(model_name, rti=True, filename='exp_config.yaml')
    params.build = args['build']
    params.act = args['activation']
    model = AdamModel(params, n_dofs=6)
    # Define the reduced model for the safe abort
    model_red = AdamModel(params, n_dofs=params.nn_dofs)
    nq = model.nq
    model.ee_ref = ee_ref
    model.R_ref = R_ref

    kp, kd = 0.1, 1e2

    cont_name = args['controller']
    ocp = get_ocp(cont_name, model, obstacles, capsules, capsule_pairs)
    opti = ocp.opti
    # Options for the initial guess
    opts = {
            'ipopt.print_level': 5,
            'print_time': 0,
            'ipopt.tol': 1e-6,
            'ipopt.constr_viol_tol': 1e-6,
            'ipopt.compl_inf_tol': 1e-6,
            'ipopt.hessian_approximation': 'limited-memory',
            'ipopt.max_iter': params.nlp_max_iter
            }
    opti.solver('ipopt', opts)  
    controller = get_controller(cont_name, model, obstacles, capsules, capsule_pairs)
    if args['build']:
        print('*** Ready for running the MPC at the next launch ***')
        exit()
    safe_ocp = SafeBackupController(model, obstacles, capsules, capsule_pairs)

    q0 = np.array([0., 0.26178, -0.26178, 0., 0., 0.])
    q0 = q0[:nq]
    x0 = np.zeros(model.nx)
    x0[:nq] = q0

    # Initial guess
    print('\n', '*'*5, 'WARM START', '*'*5, '\n')
    N = params.N
    opti.set_value(ocp.x_init, x0)
    for k in range(N):
        opti.set_initial(ocp.X[k], x0)
        opti.set_initial(ocp.U[k], np.zeros(model.nu))
    opti.set_initial(ocp.X[-1], x0)

    try:
        sol = opti.solve()
        xg = np.array([sol.value(ocp.X[k]) for k in range(params.N + 1)])
        ug = np.array([sol.value(ocp.U[k]) for k in range(params.N)])
    except:
        sol = opti.debug
        exit()

    try:
        print(f'NN on last state: {model.nn_func(xg[-1], params.alpha)}')
    except:
        print('No NN, no party')

    queue = mp.Queue(maxsize=1)  

    # Create processes
    mpc_process = mp.Process(target=run_mpc, args=(queue, xg, ug))
    viz_process = mp.Process(target=run_visualizer, args=(queue,))

    # Start processes
    mpc_process.start()
    viz_process.start()

    # Wait until MPC finish
    mpc_process.join()

    # Terminate the visualizer
    viz_process.terminate()
    viz_process.join()

    print('*** END ***')
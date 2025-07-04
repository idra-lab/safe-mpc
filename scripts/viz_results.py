import pickle
import time
import numpy as np
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.utils import rot_mat_x, rot_mat_y, rot_mat_z 
from safe_mpc.robot_visualizer import RobotVisualizer
from safe_mpc.env_model import AdamModel
from safe_mpc.cost_definition import *
from safe_mpc.controller import NaiveController, AbstractController, RecedingController
import meshcat

args = parse_args()
RENDER_CAPS = True
params = Parameters(args,'z1', True)
params.build = False
model = AdamModel(params)
#cost = Tracking8NLS(model,params.Q_weight,params.R_weight)
cost = ReachTargetNLS(model,params.Q_weight,params.R_weight)
robot = NaiveController(model)
robot.track_traj = False
rviz = RobotVisualizer(params, robot.model.params.nq)
if not(robot.track_traj):
    rviz.setTarget(params.ee_ref)

if robot.model.params.obstacles != None:
    rviz.addObstacles(robot.model.params.obstacles)

data = pickle.load(open(f'{params.DATA_DIR}z1_parallel2_use_netTrue_20hor_20sm_noise_2.5_control_noise0.0_q_collision_margins_0.1_0.001_mpc.pkl', 'rb'))

#data = pickle.load(open(f'{params.DATA_DIR}x_traj_opt.pkl','rb'))
#x=data

x = data['x']
#x=x.reshape((1,x.shape[0],x.shape[1]))
time.sleep(1)
for j in range(0,params.test_num):
    print(f"Trajectory {j + 1}")
    rviz.display(x[j][0, :model.nq])
    time.sleep(1)
    if robot.track_traj:
        rviz.addTraj(cost.traj[:,0:params.N*3:30])
        rviz.vizTraj(cost.traj[:,0:params.N*3:30])
    else:
      rviz.setTarget(params.ee_ref)
    if robot.model.params.robot_capsules != None:
            rviz.init_capsule(robot.model.params.robot_capsules)
    rviz.init_spheres(robot.model.params.spheres_robot)
    if robot.model.params.obst_capsules != None:
            rviz.init_capsule(robot.model.params.obst_capsules)  

    for i in range(params.n_steps):
        print(f'EE position: {model.ee_fun(x[j][i])}')
        if np.isnan(x[j][i,0]):
            print(i)
            print(x[j][i])
            break
        if robot.track_traj:
            rviz.vizTraj(cost.traj[:,i:i+params.N*3:30])
            if (i%100)==0:
                print(i)
                # vel=(robot.acc_traj*i)
                # if vel <= robot.vel_max_traj:
                #     print(f'Trajectory velocity at step {i} : {vel} [m/s]')
        if RENDER_CAPS:
            rviz.displayWithEESphere(x[j][i, :model.nq],robot.model.params.robot_capsules+robot.model.params.obst_capsules,robot.model.params.spheres_robot)
        else:
             T_ee = np.eye(4)
             T_ee[:3,3] = np.array(robot.model.ee_fun_noisy(x[j][i])).reshape(3)
             rviz.displayWithEE(x[j][i, :model.nq],T_ee)

        #time.sleep(0.01)
import pickle
import time
import numpy as np
from safe_mpc.parser import Parameters
from safe_mpc.utils import rot_mat_x, rot_mat_y, rot_mat_z 
from safe_mpc.robot_visualizer import RobotVisualizer
from safe_mpc.env_model import AdamModel
from safe_mpc.controller import NaiveController, AbstractController, RecedingController
import meshcat

def get_x_from_theta(theta,a):
    return (a*np.cos(theta))/(1+np.sin(theta)**2)
def get_y_from_theta(theta,a):
    return (a*np.cos(theta)*np.sin(theta))/(1+np.sin(theta)**2)

params = Parameters('z1', True)
params.build = False
model = AdamModel(params)
robot = RecedingController(model)
robot.track_traj = True
rviz = RobotVisualizer(params, 4)
if not(robot.track_traj):
    rviz.setTarget(params.ee_ref)

if robot.model.params.obstacles != None:
    rviz.addObstacles(robot.model.params.obstacles)

if robot.track_traj:
    theta = np.linspace(0,2*np.pi,100)

    theta_rot = params.theta_rot_traj
    rot_mat = rot_mat_x(theta_rot[0])@rot_mat_y(theta_rot[1])@rot_mat_z(theta_rot[2])



    x= get_x_from_theta(theta,params.dim_shape_8)
    y= get_y_from_theta(theta,params.dim_shape_8)
    z=np.zeros(x.shape[0])

    x_trj = np.vstack((x,y,z))
    x_trj=rot_mat[:3,:3]@x_trj + params.offset_traj.reshape((3,1))

data = pickle.load(open(f'{params.DATA_DIR}z1_receding_use_netTrue_40hor_10sm_traj_trackmpc.pkl', 'rb'))

x = data['x']

time.sleep(1)
for j in range(0,params.test_num if not(robot.track_traj) else 1):
    print(f"Trajectory {j + 1}")
    rviz.display(x[j][0, :model.nq])
    time.sleep(1)
    print(robot.track_traj)
    if robot.track_traj:
        rviz.addTraj(x_trj)
        rviz.vizTraj(x_trj)
    else:
      rviz.setTarget(params.ee_ref)
    if robot.model.params.robot_capsules != None:
            rviz.init_capsule(robot.model.params.robot_capsules)
    rviz.init_spheres(robot.model.params.spheres_robot)
    if robot.model.params.obst_capsules != None:
            rviz.init_capsule(robot.model.params.obst_capsules)  

    for i in range(params.n_steps):
        if np.isnan(x[j][i,0]):
            print(i)
            print(x[j][i])
            break
        if robot.track_traj:
            if (i%100)==0:
                print(i)
                # vel=(robot.acc_traj*i)
                # if vel <= robot.vel_max_traj:
                #     print(f'Trajectory velocity at step {i} : {vel} [m/s]')
        if params.use_capsules:
            rviz.displayWithEESphere(x[j][i, :model.nq],robot.model.params.robot_capsules+robot.model.params.obst_capsules,robot.model.params.spheres_robot)
        else:
             T_ee = np.eye(4)
             T_ee[:3,3] = np.array(robot.model.ee_fun_noisy(x[j][i])).reshape(3)
             rviz.displayWithEE(x[j][i, :model.nq],T_ee)

        #time.sleep(params.dt)
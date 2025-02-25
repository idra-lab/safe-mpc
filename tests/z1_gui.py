import time
import threading
import numpy as np
import tkinter as tk
import pinocchio as pin
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.utils import obstacles, ee_ref, RobotVisualizer, capsules, capsule_pairs
from safe_mpc.abstract import AdamModel
from safe_mpc.controller import NaiveController


axes = ['x', 'y', 'z']


class ScaleJoints:
    def __init__(self, master, name, nq, from_, to, tickinterval, length, command):
        self.s = nq * [None]
        for i in range(nq):
            self.s[i] = tk.Scale(master, label=f'{name} {i + 1}', from_=from_[i], to=to[i], orient=tk.HORIZONTAL, 
                                 tickinterval=tickinterval, resolution=1e-3, length=length, command=command)
            self.s[i].pack()
        separator = tk.Frame(master, height=2, bd=1, relief=tk.SUNKEN)
        separator.pack(fill=tk.X, padx=5, pady=5)

    def get(self):
        return [s.get() for s in self.s]
    

class PositionEE:
    def __init__(self, master):
        self.l = 3 * [None]
        for i in range(3):
            self.l[i] = tk.Label(master, text=f'Position {axes[i]}: 0.0')
            self.l[i].pack()


scale_joints = None
position_ee = None
q = None


def update_ee(value):
    global q
    q = np.array(scale_joints.get())
    pin.forwardKinematics(rmodel_red, rdata, q)
    pin.updateFramePlacement(rmodel_red, rdata, frame_id)
    ee_trasl = rdata.oMf[frame_id].translation
    position_ee.l[0].config(text=f'Position {axes[0]}: {ee_trasl[0]:.3f}')
    position_ee.l[1].config(text=f'Position {axes[1]}: {ee_trasl[1]:.3f}')
    position_ee.l[2].config(text=f'Position {axes[2]}: {ee_trasl[2]:.3f}')


def create_gui():
    global scale_joints, position_ee
    master = tk.Tk(className='Z1 Robot GUI')
    scale_joints = ScaleJoints(master, 'Joint', model.nq, 
                               rmodel_red.lowerPositionLimit, 
                               rmodel_red.upperPositionLimit, 
                               np.pi / 2, 300, update_ee)
    position_ee = PositionEE(master)
    master.mainloop()

def run_visualizer():
    global q
    i = 0
    dt = 1e-3
    display_n = 10
    rviz.displayWithEESphere(np.zeros(model.nq), robot.capsules)
    while True:
        time_start = time.time()
        if i % display_n == 0 and q is not None:
            rviz.displayWithEESphere(q, robot.capsules)
        i += 1
        time_end = time.time()
        time.sleep(max(0, dt - (time_end - time_start)))
    

args = parse_args()
nq = args['dofs']
cont_name = args['controller']
alpha = args['alpha']
horizon = args['horizon']
params = Parameters('z1', True)
model = AdamModel(params, nq)
model.ee_ref = ee_ref
robot = NaiveController(model, obstacles, capsules, capsule_pairs)
rviz = RobotVisualizer(params, nq)
rviz.setTarget(ee_ref)
if params.obs_flag:
    rviz.addObstacles(obstacles)
for capsule in robot.capsules:
    rviz.init_capsule(capsule)

description_dir = params.ROBOTS_DIR + 'z1_description'
rmodel, collision, visual = pin.buildModelsFromUrdf(description_dir + '/urdf/z1.urdf',
                                                    package_dirs=params.ROOT_DIR)
geom = [collision, visual]

lockIDs = []
joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'jointGripper']
lockNames = joint_names[nq:]
for name in lockNames:
    lockIDs.append(rmodel.getJointId(name))

rmodel_red, geom_red = pin.buildReducedModel(rmodel, geom, lockIDs, np.zeros(7))
rdata = rmodel_red.createData()
frame_id = rmodel_red.getFrameId(params.frame_name)

th_gui = threading.Thread(target=create_gui)
th_gui.start()

th_simu = threading.Thread(target=run_visualizer)
th_simu.start()

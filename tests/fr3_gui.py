import time
import threading
import numpy as np
import tkinter as tk
import pinocchio 
from safe_mpc.parser import Parameters, parse_args
import meshcat

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
    pinocchio.forwardKinematics(rmodel_red, rdata, q)
    pinocchio.updateFramePlacement(rmodel_red, rdata, frame_id)
    ee_trasl = rdata.oMf[frame_id].translation
    position_ee.l[0].config(text=f'Position {axes[0]}: {ee_trasl[0]:.3f}')
    position_ee.l[1].config(text=f'Position {axes[1]}: {ee_trasl[1]:.3f}')
    position_ee.l[2].config(text=f'Position {axes[2]}: {ee_trasl[2]:.3f}')


def create_gui():
    global scale_joints, position_ee
    master = tk.Tk(className='Z1 Robot GUI')
    scale_joints = ScaleJoints(master, 'Joint', n_dofs, 
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
    while True:
        time_start = time.time()
        if i % display_n == 0:
            viz.display(q)
        time_end = time.time()
        time.sleep(max(0, dt - (time_end - time_start)))

args = parse_args()
params = Parameters(args,'fr3')
n_dofs = 7
# robot = adam_model.AdamModel(params,n_dofs=n_dofs)
description_dir = params.ROBOTS_DIR + 'fr3_description'
rmodel, collision, visual = pinocchio.buildModelsFromUrdf(description_dir + '/urdf/fr3.urdf',
                                                    package_dirs=params.ROOT_DIR)
geom = [collision, visual]

lockIDs = []
#lockNames = ['joint5', 'joint6', 'jointGripper']
lockNames = ['fr3_finger_joint1','fr3_finger_joint2']
# lockNames = lockNames[(n_dofs-4):]
for name in lockNames:
    lockIDs.append(rmodel.getJointId(name))

rmodel_red, geom_red = pinocchio.buildReducedModel(rmodel, geom, lockIDs, np.zeros(9))
rdata = rmodel_red.createData()
viz = pinocchio.visualize.MeshcatVisualizer(rmodel_red, geom_red[0], geom_red[1])
viz.initViewer(loadModel=True, open=True)
# viz.setCameraPosition(np.array([0.5, -0.5, 0.4]))
# viz.setCameraTarget(np.array([0., 1, 0.]))
viz.display(np.zeros(rmodel_red.nq))
frame_id = rmodel_red.getFrameId(params.frame_name)

# box0 = meshcat.geometry.Box([2, 2, 1e-3])
# viz.viewer['world/obstacle/floor0'].set_object(box0)
# viz.viewer['world/obstacle/floor0'].set_property('color', [0, 0, 1, 0.5])
# viz.viewer['world/obstacle/floor0'].set_property('visible', True)
# T_floor = np.eye(4)
# viz.viewer['world/obstacle/floor0'].set_transform(T_floor)

# box1 = meshcat.geometry.Box([2, 2, 1e-3])
# viz.viewer['world/obstacle/floor1'].set_object(box1)
# viz.viewer['world/obstacle/floor1'].set_property('color', [0, 0, 1, 0.5])
# viz.viewer['world/obstacle/floor1'].set_property('visible', True)
# T_floor = np.eye(4)
# T_floor[:3,3] = np.array([0,0,0.6])
# viz.viewer['world/obstacle/floor1'].set_transform(T_floor)


# box2 = meshcat.geometry.Box([1e-3, 2, 2])
# viz.viewer['world/obstacle/floor2'].set_object(box2)
# viz.viewer['world/obstacle/floor2'].set_property('color', [0, 0, 1, 0.5])
# viz.viewer['world/obstacle/floor2'].set_property('visible', True)
# T_floor = np.eye(4)
# T_floor[:3,3] = np.array([0.5,0,0])
# viz.viewer['world/obstacle/floor2'].set_transform(T_floor)

# sphere = meshcat.geometry.Sphere(0.12)
# viz.viewer['world/obstacle/sphere'].set_object(sphere)
# viz.viewer['world/obstacle/sphere'].set_property('color',[1, 0, 0, 1])
# viz.viewer['world/obstacle/sphere'].set_property('visible', True)
# T_obs = np.eye(4)
# T_obs[:3, 3] = np.array([0.6,0.,0.12])
# viz.viewer['world/obstacle/sphere'].set_transform(T_obs)

# sphere2 = meshcat.geometry.Sphere(0.1)
# viz.viewer['world/obstacle/sphere2'].set_object(sphere2)
# viz.viewer['world/obstacle/sphere2'].set_property('color',[1, 0, 0, 1])
# viz.viewer['world/obstacle/sphere2'].set_property('visible', True)
# T_obs = np.eye(4)
# T_obs[:3, 3] = np.array([0.3,-0.35,0.25])
# viz.viewer['world/obstacle/sphere2'].set_transform(T_obs)

# sphere = meshcat.geometry.Sphere(0.05)
# viz.viewer['world/obstacle/sphere'].set_object(sphere)
# viz.viewer['world/obstacle/sphere'].set_property('color',[1, 0, 0, 1])
# viz.viewer['world/obstacle/sphere'].set_property('visible', True)
# T_obs = np.eye(4)
# T_obs[:3, 3] = np.array([0.,0.3,0.15])
# viz.viewer['world/obstacle/sphere'].set_transform(T_obs)

th_gui = threading.Thread(target=create_gui)
th_gui.start()

th_simu = threading.Thread(target=run_visualizer)
th_simu.start()
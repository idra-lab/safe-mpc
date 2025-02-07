import time 
import meshcat
import numpy as np
import pinocchio as pin
from .abstract import AbstractController
from .controller import *
from .ocp import *


### OBSTACLES ###
ee_radius = 0.075
ee_ref = np.array([0.6, 0.28, 0.078])

obs = dict()
obs['name'] = 'floor'
obs['type'] = 'box'
obs['dimensions'] = [2, 2, 1e-3]
obs['color'] = [0, 0, 1, 1]
obs['position'] = np.array([0., 0., 0.])
obs['transform'] = np.eye(4)
obs['bounds'] = np.array([ee_radius, 1e6])      # lb , ub
obstacles = [obs]

obs = dict()
obs['name'] = 'ball'
obs['type'] = 'sphere'
obs['radius'] = 0.12
obs['color'] = [0, 1, 1, 1]
obs['position'] = np.array([0.6, 0., 0.12])
T_ball = np.eye(4)
T_ball[:3, 3] = obs['position']
obs['transform'] = T_ball
obs['bounds'] = np.array([(ee_radius + obs['radius']) ** 2, 1e6])     
obstacles.append(obs)


### METHODS ###

def get_ocp(ocp_name, model, obstacles) -> NaiveOCP:
    ocps = { 'naive': NaiveOCP,
             'zerovel': TerminalZeroVelOCP,
             'st': SoftTerminalOCP,
             'htwa': SoftTerminalOCP,
             'receding': SoftTerminalOCP, 
             'real': SoftTerminalOCP}
    if ocp_name in ocps:
        return ocps[ocp_name](model, obstacles)
    else:
        raise ValueError(f'OCP {ocp_name} not available')

def get_controller(cont_name, model, obstacles) -> AbstractController:
    controllers = { 'naive': NaiveController,
                    'zerovel': TerminalZeroVelocity,
                    'st': STController,
                    'htwa': HTWAController,
                    'receding': RecedingController,
                    'real': RealReceding }
    if cont_name in controllers:
        return controllers[cont_name](model, obstacles)
    else:
        raise ValueError(f'Controller {cont_name} not available')


### VISUALIZER ###
class RobotVisualizer:
    def __init__(self, params, n_dofs=4):
        self.params = params
        description_dir = params.ROBOTS_DIR + 'z1_description'
        rmodel, collision, visual = pin.buildModelsFromUrdf(description_dir + '/urdf/z1.urdf',
                                                            package_dirs=params.ROOT_DIR)
        geom = [collision, visual]
        lockIDs = []
        lockNames = rmodel.names.tolist()[1:]               # skip 'universe' joint 
        for name in lockNames[n_dofs:]:
            lockIDs.append(rmodel.getJointId(name))
        rmodel_red, geom_red = pin.buildReducedModel(rmodel, geom, lockIDs, np.zeros(rmodel.nq))   
        
        self.viz = pin.visualize.MeshcatVisualizer(rmodel_red, geom_red[0], geom_red[1])
        self.viz.initViewer(loadModel=True, open=True)
        # self.viz.setCameraPosition(np.array([1., 1., 1.]))

        # Set the end-effector target
        ee_radius = 0.075   
        shpere = meshcat.geometry.Sphere(ee_radius)
        self.viz.viewer['world/robot/target'].set_object(shpere)
        self.viz.viewer['world/robot/target'].set_property('color', [0, 1, 0, 0.4])
        self.viz.viewer['world/robot/target'].set_property('visible', True)
        T_target = np.eye(4)
        self.viz.viewer['world/robot/target'].set_transform(T_target)
        # EE
        self.viz.viewer['world/robot/ee'].set_object(shpere)
        self.viz.viewer['world/robot/ee'].set_property('color', [1, 1, 0, 0.7])
        self.viz.viewer['world/robot/ee'].set_property('visible', False)
        self.viz.viewer['world/robot/ee'].set_transform(T_target)

    def setTarget(self, ee_ref):
        T_target = np.eye(4)
        T_target[:3, 3] = ee_ref
        self.viz.viewer['world/robot/target'].set_transform(T_target)

    def setInitialBox(self):
        dims = self.params.box_ub - self.params.box_lb
        pos = (self.params.box_ub + self.params.box_lb) / 2
        box = meshcat.geometry.Box(dims)
        self.viz.viewer['world/obstacle/box'].set_object(box)
        self.viz.viewer['world/obstacle/box'].set_property('color', [1, 0, 0, 0.3])
        self.viz.viewer['world/obstacle/box'].set_property('visible', True)
        T_box = np.eye(4)
        T_box[:3, 3] = pos
        self.viz.viewer['world/obstacle/box'].set_transform(T_box)
    
    def addObstacles(self, obstacles):
        for obs in obstacles:
            if obs['type'] == 'box':
                box = meshcat.geometry.Box(obs['dimensions'])
                self.viz.viewer['world/obstacle/' + obs['name']].set_object(box)
                self.viz.viewer['world/obstacle/' + obs['name']].set_property('color', obs['color'])
                self.viz.viewer['world/obstacle/' + obs['name']].set_property('visible', True)
                T_obs = np.eye(4)
                T_obs[:3, 3] = obs['position']
                self.viz.viewer['world/obstacle/' + obs['name']].set_transform(T_obs)
            elif obs['type'] == 'sphere':
                sphere = meshcat.geometry.Sphere(obs['radius'])
                self.viz.viewer['world/obstacle/' + obs['name']].set_object(sphere)
                self.viz.viewer['world/obstacle/' + obs['name']].set_property('color', obs['color'])
                self.viz.viewer['world/obstacle/' + obs['name']].set_property('visible', True)
                T_obs = np.eye(4)
                T_obs[:3, 3] = obs['position']
                self.viz.viewer['world/obstacle/' + obs['name']].set_transform(T_obs)

    def display(self, q):
        self.viz.display(q)
        time.sleep(self.params.dt)

    def displayWithEE(self, q, T_ee):
        self.viz.display(q)
        self.viz.viewer['world/robot/ee'].set_property('visible', True)
        self.viz.viewer['world/robot/ee'].set_transform(T_ee)
        time.sleep(self.params.dt)

    def moveCamera(self, cam_pos):
        self.viz.setCameraPosition(cam_pos)

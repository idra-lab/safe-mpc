import time 
import copy
import meshcat
import numpy as np
import pinocchio as pin
from .controller import AbstractController
from .controller import *
from .ocp import *
from urdf_parser_py.urdf import URDF
from .parser import Parameters, parse_args
import xml.etree.ElementTree as ET

class RobotVisualizer:
    def __init__(self, params, n_dofs=4):
        self.params = params
        rmodel, collision, visual = pin.buildModelsFromUrdf(self.params.robot_urdf,
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
        ee_radius = params.ee_radius   
        sphere = meshcat.geometry.Sphere(ee_radius)
        self.viz.viewer['world/robot/target'].set_object(sphere)
        self.viz.viewer['world/robot/target'].set_property('color', [0, 1, 0, 0.4])
        self.viz.viewer['world/robot/target'].set_property('visible', False)
        T_target = np.eye(4)
        self.viz.viewer['world/robot/target'].set_transform(T_target)
        # EE
        self.viz.viewer['world/robot/ee'].set_object(sphere)
        self.viz.viewer['world/robot/ee'].set_property('color', [1, 1, 0, 0.7])
        self.viz.viewer['world/robot/ee'].set_property('visible', False)
        self.viz.viewer['world/robot/ee'].set_transform(T_target)

    def setTarget(self, ee_ref):
        T_target = np.eye(4)
        T_target[:3, 3] = ee_ref
        self.viz.viewer['world/robot/target'].set_transform(T_target)
        self.viz.viewer['world/robot/target'].set_property('visible', True)

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
            if obs['type'] == 'plane':
                box = meshcat.geometry.Box(obs['dimensions'])
                self.viz.viewer['world/obstacle/' + obs['name']].set_object(box)
                self.viz.viewer['world/obstacle/' + obs['name']].set_property('color', list(obs['color']))
                self.viz.viewer['world/obstacle/' + obs['name']].set_property('visible', True)
                T_obs = np.eye(4)
                T_obs[:3, 3] = obs['position']
                self.viz.viewer['world/obstacle/' + obs['name']].set_transform(T_obs)
            elif obs['type'] == 'sphere':
                sphere = meshcat.geometry.Sphere(obs['radius'])
                self.viz.viewer['world/obstacle/' + obs['name']].set_object(sphere)
                self.viz.viewer['world/obstacle/' + obs['name']].set_property('color', list(obs['color']))
                self.viz.viewer['world/obstacle/' + obs['name']].set_property('visible', True)
                T_obs = np.eye(4)
                T_obs[:3, 3] = obs['position']
                self.viz.viewer['world/obstacle/' + obs['name']].set_transform(T_obs)

    def init_capsule(self,capsules):
        for capsule in capsules:
            sphere1 = meshcat.geometry.Sphere(capsule['radius'])
            self.viz.viewer[f'world/obstacle/sphere1{capsule["index"]}'].set_object(sphere1)
            self.viz.viewer[f'world/obstacle/sphere1{capsule["index"]}'].set_property('color', capsule['color'])
            self.viz.viewer[f'world/obstacle/sphere1{capsule["index"]}'].set_property('visible', True)
            self.viz.viewer[f'world/obstacle/sphere1{capsule["index"]}'].set_transform(np.eye(4))
            cylinder = meshcat.geometry.Cylinder(capsule['length'],capsule['radius'])
            self.viz.viewer[f'world/obstacle/cylinder1{capsule["index"]}'].set_object(cylinder)
            self.viz.viewer[f'world/obstacle/cylinder1{capsule["index"]}'].set_property('color', capsule['color'])
            self.viz.viewer[f'world/obstacle/cylinder1{capsule["index"]}'].set_property('visible', True)
            self.viz.viewer[f'world/obstacle/cylinder1{capsule["index"]}'].set_transform(np.eye(4))
            sphere2 = meshcat.geometry.Sphere(capsule['radius'])
            self.viz.viewer[f'world/obstacle/sphere2{capsule["index"]}'].set_object(sphere2)
            self.viz.viewer[f'world/obstacle/sphere2{capsule["index"]}'].set_property('color', capsule['color'])
            self.viz.viewer[f'world/obstacle/sphere2{capsule["index"]}'].set_property('visible', True)
            self.viz.viewer[f'world/obstacle/sphere2{capsule["index"]}'].set_transform(np.eye(4))

    def render_capsule(self,q,capsule):
        rot_mat = np.array([[0,1,0,0],
                            [1,0,0,0],
                            [0,0,1,0],
                            [0,0,0,1]])
        # self.viz.viewer[f'world/obstacle/sphere1{capsule["index"]}'].set_property('visible', True)
        if capsule['type'] == 'moving_capsule':
            self.viz.viewer[f'world/obstacle/sphere1{capsule["index"]}'].set_transform(np.array(self.compute_T(capsule['end_points_T_fun'](q),capsule['end_points'][0])))
            self.viz.viewer[f'world/obstacle/cylinder1{capsule["index"]}'].set_transform(np.array(self.compute_T(capsule['end_points_T_fun'](q),(capsule['end_points'][0]+capsule['end_points'][1])/2)@rot_mat))
            self.viz.viewer[f'world/obstacle/sphere2{capsule["index"]}'].set_transform(np.array(self.compute_T(capsule['end_points_T_fun'](q),capsule['end_points'][1])))
        elif capsule['type'] == 'fixed_capsule':
            T_Tmp = np.eye(4)
            T_Tmp[:3,3] = capsule['end_points'][0]
            self.viz.viewer[f'world/obstacle/sphere1{capsule["index"]}'].set_transform(T_Tmp)
            T_Tmp = np.eye(4)
            T_Tmp[:3,:3] = capsule['end_points_T_fun']
            T_Tmp[:3,3] = (capsule['end_points'][0]+capsule['end_points'][1])/2
            self.viz.viewer[f'world/obstacle/cylinder1{capsule["index"]}'].set_transform(T_Tmp)
            T_Tmp = np.eye(4)
            T_Tmp[:3,3] = capsule['end_points'][1]
            self.viz.viewer[f'world/obstacle/sphere2{capsule["index"]}'].set_transform(T_Tmp)

    def display(self, q):
        self.viz.display(q)
        time.sleep(self.params.dt)

    def displayWithEE(self, q, T_ee):
        self.viz.display(q)
        self.viz.viewer['world/robot/ee'].set_property('visible', True)
        self.viz.viewer['world/robot/ee'].set_transform(T_ee)
        time.sleep(self.params.dt)

    def compute_T(self,T,vec):
        res = copy.deepcopy(T)
        res[:3,3]= (res @ vec)[:3] 
        return res

    def displayWithEESphere(self, q, capsules):
        self.viz.display(q)
        q = np.hstack((q,np.zeros(q.shape[0])))
        for capsule in capsules:
            self.render_capsule(q,capsule)

    def addTraj(self,points):
        for i in range(points.shape[1]):
            sphere = meshcat.geometry.Sphere(0.01)
            self.viz.viewer['world/obstacle/' + f'traj_point{i}'].set_object(sphere)
            self.viz.viewer['world/obstacle/' + f'traj_point{i}'].set_property('color', [1,0,0,1])
            self.viz.viewer['world/obstacle/' + f'traj_point{i}'].set_property('visible', True)
            T_obs = np.eye(4)
            T_obs[:3, 3] = points[:,i]
            self.viz.viewer['world/obstacle/' + f'traj_point{i}'].set_transform(T_obs)

    def vizTraj(self,points): 
        T_obs = np.eye(4)
        for i in range(points.shape[1]):
            self.viz.viewer['world/obstacle/' + f'traj_point{i}'].set_property('visible', True)
            T_obs[:3, 3] = points[:,i]
            self.viz.viewer['world/obstacle/' + f'traj_point{i}'].set_transform(T_obs)

    def moveCamera(self, cam_pos):
        self.viz.setCameraPosition(cam_pos)
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

args = parse_args()
model_name = args['system']
params = Parameters(model_name)


### METHODS ###

def get_ocp(ocp_name, model) -> NaiveOCP:
    ocps = { 'naive': NaiveOCP,
             'zerovel': TerminalZeroVelOCP,
             'st': SoftTerminalOCP,
             'htwa': HardTerminalOCP,
             'receding': HardTerminalOCP,
             'parallel': HardTerminalOCP, 
             'real': HardTerminalOCP}
    if ocp_name in ocps:
        return ocps[ocp_name](model)
    else:
        raise ValueError(f'OCP {ocp_name} not available')
    
def get_ocp_acados(cont_name, model) -> AbstractController:
    controllers = { 'naive': NaiveController,
                    'zerovel': TerminalZeroVelocity,
                    'st': HTWAController,
                    'htwa': HTWAController,
                    'receding': HTWAController,
                    'parallel': HTWAController,
                    'st_analytic': HTWAController,
                    'htwa_analytic': HTWAController,
                    'receding_analytic': HTWAController,
                    'parallel_analytic': HTWAController}
    if cont_name in controllers:
        return controllers[cont_name](model), controllers
    else:
        raise ValueError(f'Controller {cont_name} not available')

def get_controller(cont_name, model) -> AbstractController:
    controllers = { 'naive': NaiveController,
                    'zerovel': TerminalZeroVelocity,
                    'st': STController,
                    'htwa': HTWAController,
                    'receding': RecedingController,
                    'parallel': ParallelController,
                    'real': RealReceding }
    if cont_name in controllers:
        return controllers[cont_name](model)
    else:
        raise ValueError(f'Controller {cont_name} not available')
    
def rot_mat_x(theta):
    return np.array([[1,0,0,0],
                     [0,np.cos(theta),-np.sin(theta),0],
                     [0,np.sin(theta),np.cos(theta),0],
                     [0,0,0,1]])
    
def rot_mat_y(theta):
    return  np.array([[np.cos(theta),0,np.sin(theta),0],
                      [0,1,0,0],
                      [-np.sin(theta),0,np.cos(theta),0],
                      [0,0,0,1]])
def rot_mat_z(theta):
    return np.array([[np.cos(theta),-np.sin(theta),0,0],
                     [np.sin(theta), np.cos(theta),0,0],
                     [0,0,1,0],
                     [0,0,0,1]])

def casadi_segment_dist(A_s,B_s,C_s,D_s):

        R = cs.sum1((B_s-A_s)*(D_s-C_s))
        S1 = cs.sum1((B_s-A_s)*(C_s-A_s))
        D1 = cs.sum1((B_s-A_s)**2)
        S2 = cs.sum1((D_s-C_s)*(C_s-A_s))
        D2 = cs.sum1((D_s-C_s)**2)

        t = (S1*D2 - S2*R)/(D1*D2 - (R**2+1e-5))
        t = cs.fmax(cs.fmin(t,1),0)

        u = (t*R - S2)/D2
        u = cs.fmax(cs.fmin(u,1),0)

        t = (u*R + S1) / D1
        t = cs.fmax(cs.fmin(t,1),0)

        constr_expr = cs.sum1(((B_s-A_s)*t - (D_s-C_s)*u - (C_s-A_s))**2)

        return constr_expr
    
def ball_segment_dist(A_s,B_s,capsule_length,obs_pos):
    t = cs.fmin(cs.fmax(cs.dot((obs_pos-A_s),(B_s-A_s)) / (capsule_length**2),0),1)
    d = cs.sum1((obs_pos-(A_s+(B_s-A_s)*t))**2) 
    return d

def ball_ee_dist(obs,ee_expr):
    return (ee_expr - obs['position']).T @ (ee_expr - obs['position'])

def floor_ee_dist(obs,ee_expr):
    return ee_expr[2] - obs['bounds'][0]

def randomize_model(urdf_file_path,noise_mass_percentage=0, noise_inertia_percentage=0, noise_cm_position_percentage=0):
    inertia_fields = ['ixx','iyy','izz', 'ixy', 'iyz' , 'ixz']
    # Load the URDF file
    tree = ET.parse(urdf_file_path)
    root = tree.getroot()
    links = root.findall('link')
    for link in links:
        inertial = link.find('inertial')
        if inertial is not None:
            mass=inertial.find('mass')
            noise = float(mass.get('value')) * noise_mass_percentage
            new_mass = float(mass.get('value')) + np.random.uniform(-noise, noise)
            mass.set('value', str(new_mass))
            
            inertia = inertial.find('inertia')
            for i in inertia_fields:
                noise =  abs(float(inertia.get(i))) * noise_inertia_percentage
                new_inertia = float(inertia.get(i))+np.random.uniform(-noise, noise)
                inertia.set(i,str(new_inertia))
            
            cm_pos = inertial.find('origin')
            pos = list(map(float, cm_pos.get('xyz').split(' ')))
            for e in pos:
                noise = abs(e*noise_cm_position_percentage)
                e += np.random.uniform(-noise, noise)
            cm_pos.set('xyz', ' '.join(map(str, pos)))

    # Write the modified URDF back to a file
    tree.write(urdf_file_path[:-5] + '_randomized.urdf', encoding='utf-8', xml_declaration=True)

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
        ee_radius = params.ee_radius   
        shpere = meshcat.geometry.Sphere(ee_radius)
        self.viz.viewer['world/robot/target'].set_object(shpere)
        self.viz.viewer['world/robot/target'].set_property('color', [0, 1, 0, 0.4])
        self.viz.viewer['world/robot/target'].set_property('visible', False)
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

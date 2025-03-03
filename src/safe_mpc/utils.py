import time 
import copy
import meshcat
import numpy as np
import pinocchio as pin
from .abstract import AbstractController
from .controller import *
from .ocp import *
from urdf_parser_py.urdf import URDF
from .parser import Parameters, parse_args


args = parse_args()
model_name = args['system']
params = Parameters(model_name)
robot = URDF.from_xml_file(params.robot_urdf)
links = [robot.links[i].name for i in range(len(robot.links))]
joints = [robot.joints[i] for i in range(len(robot.joints))]


def align_vectors(a, b):
    b = b / np.linalg.norm(b) # normalize a
    a = a / np.linalg.norm(a) # normalize b
    v = np.cross(a, b)
    # s = np.linalg.norm(v)
    c = np.dot(a, b)
    if np.isclose(c, -1.0):
        return -np.eye(3, dtype=np.float64)

    v1, v2, v3 = v
    h = 1 / (1 + c)

    Vmat = np.array([[0, -v3, v2],
                  [v3, 0, -v1],
                  [-v2, v1, 0]])

    R = np.eye(3, dtype=np.float64) + Vmat + (Vmat.dot(Vmat) * h)
    return R

def create_moving_capsule(name: str,link_name:str,link_axis: int,radius: float,length: float,spatial_offset,rotation_offset=None,color=[1,0,0,0.3]):
    capsule=dict()
    capsule['type'] = 'moving'
    capsule['axis'] = link_axis
    capsule['radius'] = radius
    capsule['length'] = length
    capsule['name'] = name
    capsule['link_name'] = link_name
    # first point defined by offset from link origin, second length offset from the first one 
    # capsule['end_points'] = [np.hstack((spatial_offset,np.ones(1))), np.hstack((spatial_offset,np.ones(1)))]
    capsule['end_points'] = [np.hstack((np.zeros(3),np.ones(1))), np.hstack((np.zeros(3),np.ones(1)))]
    capsule['direction'] = np.sign( joints[links.index(capsule['link_name'])].origin.xyz[link_axis])
    capsule['end_points'][1][link_axis] += capsule['direction']*capsule['length']
    capsule['end_points_T_fun'] = [None]
    capsule['end_points_fk'] = [None,None]
    capsule['rotation_offset'] = rotation_offset
    capsule['spatial_offset'] = spatial_offset
    capsule['end_points_fk_fun'] = [None]
    capsule['color'] = color
    return capsule

def create_fixed_capsule(name,radius: float,fixed_A,fixed_B,color=[1,0,0,0.3]):
    capsule=dict()
    capsule['type'] = 'fixed'
    capsule['name'] = name
    capsule['end_points'] = [fixed_A,fixed_B]
    capsule['length'] = np.linalg.norm(fixed_A-fixed_B)
    capsule['radius'] = radius
    capsule['end_points_fk'] = capsule['end_points']
    capsule['end_points_T_fun'] = align_vectors(np.array([0,1,0]),capsule['end_points'][1]-capsule['end_points'][0])     
    capsule['color'] = color
    return capsule

def assign_pairs(obj1_name,obj2_name,obstacles_list,capsules_list):
    pair=dict()
    pair['elements'] = [None,None]
    pair['type'] = None
    for capsule in capsules_list:
        if obj1_name == capsule['name']:
            pair['elements'][0] = capsule
            break
    for capsule in capsules_list:
        if obj2_name == capsule['name']:
            pair['elements'][1] = capsule
            if pair['elements'][0] != None:
                pair['type'] = 0
            break 
    for obstacle in obstacles_list:
        if obj2_name == obstacle['name']:
            pair['elements'][1] = obstacle
            if obstacle['type'] == 'sphere': pair['type'] = 1
            elif obstacle['type'] == 'box': pair['type'] = 2
            break 
    return pair



### OBSTACLES ###
ee_radius = 0.05
# ee_ref = np.array([0.6, 0.28, 0.078])
ee_ref = np.array([0.7, 0., 0.078])
# ee_ref = np.array([0.5, -0.2, 0.078])

obs = dict()
obs['name'] = 'floor'
obs['type'] = 'box'
obs['dimensions'] = [1.5, 0.75, 1e-3]
obs['color'] = [0, 0, 1, 1]
obs['position'] = np.array([0.75, 0., 0.])
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
# obstacles.append(obs)

### CAPSULES ###
capsules = []

capsules.append(create_moving_capsule('arm','link02',0,0.05,0.35,[0,0,0],rotation_offset=[0,0,0]))
capsules.append(create_moving_capsule('ee','link05',0,0.05,0.22,[0,0,0],None))
#create_capsule('moving','link03',0.05,0.1,capsules,-0.0,color=[0,0,1,0.3],rotation_offset=-np.pi/6)
capsules.append(create_moving_capsule('forearm','link03',0,0.05,0.16,[0.06,0,0.05],rotation_offset=[0,-np.pi/50,0],color=[0,0,1,0.3]))
capsules.append(create_fixed_capsule('fixed1',0.025,np.array([0.5, 0.175, 0.]),np.array([0.5, 0.175, 0.225]),color=[0, 1, 1, 1]))
capsules.append(create_fixed_capsule('fixed2',0.025,np.array([0.5, -0.175, 0.]),np.array([0.5, -0.175, 0.225]),color=[0, 1, 1, 1]))
capsules.append(create_fixed_capsule('fixed3',0.025,np.array([0.5, -0.175, 0.225]),np.array([0.5, 0.175, 0.225]),color=[0, 1, 1, 1]))

# capsules.append(create_fixed_capsule('fixed1',0.01,np.array([0.3, 0.0, 0.]),np.array([0.3, 0.0, 0.25]),color=[0, 1, 1, 1]))
# capsules.append(create_fixed_capsule('fixed2',0.01,np.array([0.7, 0.0, 0.]),np.array([0.7, 0.0, 0.25]),color=[0, 1, 1, 1]))
# capsules.append(create_fixed_capsule('fixed3',0.01,np.array([0.3, 0.0, 0.25]),np.array([0.7, 0.0, 0.25]),color=[0, 1, 1, 1]))

### Pairs: CAPSULE-CAPSULE -> type 0, CAPSULE-BALL -> type 1, CAPSULE-FLOOR type 2. For now insert
# elements in the pairs in this order                                                       ###
capsule_pairs = []
capsule_pairs.append(assign_pairs('arm','ee',obstacles,capsules))
capsule_pairs.append(assign_pairs('forearm','floor',obstacles,capsules))
capsule_pairs.append(assign_pairs('ee','floor',obstacles,capsules))
capsule_pairs.append(assign_pairs('forearm','fixed1',obstacles,capsules))
capsule_pairs.append(assign_pairs('forearm','fixed2',obstacles,capsules))
capsule_pairs.append(assign_pairs('forearm','fixed3',obstacles,capsules))
capsule_pairs.append(assign_pairs('ee','fixed1',obstacles,capsules))
capsule_pairs.append(assign_pairs('ee','fixed2',obstacles,capsules))
capsule_pairs.append(assign_pairs('ee','fixed3',obstacles,capsules))


### METHODS ###

def get_ocp(ocp_name, model, obstacles, capsule, capsule_pairs) -> NaiveOCP:
    ocps = { 'naive': NaiveOCP,
             'zerovel': TerminalZeroVelOCP,
             'st': SoftTerminalOCP,
             'htwa': HardTerminalOCP,
             'receding': SoftTerminalOCP, 
             'real': SoftTerminalOCP}
    if ocp_name in ocps:
        return ocps[ocp_name](model, obstacles, capsule, capsule_pairs)
    else:
        raise ValueError(f'OCP {ocp_name} not available')

def get_controller(cont_name, model, obstacles, capsule, capsule_pairs) -> AbstractController:
    controllers = { 'naive': NaiveController,
                    'zerovel': TerminalZeroVelocity,
                    'st': STController,
                    'htwa': HTWAController,
                    'receding': RecedingController,
                    'real': RealReceding }
    if cont_name in controllers:
        return controllers[cont_name](model, obstacles, capsule, capsule_pairs)
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

    def init_capsule(self,capsule):
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
        if capsule['type'] == 'moving':
            self.viz.viewer[f'world/obstacle/sphere1{capsule["index"]}'].set_transform(np.array(self.compute_T(capsule['end_points_T_fun'](q),capsule['end_points'][0])@rot_mat))
        elif capsule['type'] == 'fixed':
            T_Tmp = np.eye(4)
            T_Tmp[:3,3] = capsule['end_points'][0]
            self.viz.viewer[f'world/obstacle/sphere1{capsule["index"]}'].set_transform(T_Tmp@rot_mat)
        # self.viz.viewer[f'world/obstacle/cylinder1{capsule["index"]}'].set_property('visible', True)
        if capsule['type'] == 'moving':
            self.viz.viewer[f'world/obstacle/cylinder1{capsule["index"]}'].set_transform(np.array(self.compute_T(capsule['end_points_T_fun'](q),(capsule['end_points'][0]+capsule['end_points'][1])/2)@rot_mat))
        elif capsule['type'] == 'fixed':
            T_Tmp = np.eye(4)
            T_Tmp[:3,:3] = capsule['end_points_T_fun']
            T_Tmp[:3,3] = (capsule['end_points'][0]+capsule['end_points'][1])/2
            self.viz.viewer[f'world/obstacle/cylinder1{capsule["index"]}'].set_transform(T_Tmp)
        # self.viz.viewer[f'world/obstacle/sphere2{capsule["index"]}'].set_property('visible', True)
        if capsule['type'] == 'moving':
            self.viz.viewer[f'world/obstacle/sphere2{capsule["index"]}'].set_transform(np.array(self.compute_T(capsule['end_points_T_fun'](q),capsule['end_points'][1])@rot_mat))
        elif capsule['type'] == 'fixed':
            T_Tmp = np.eye(4)
            T_Tmp[:3,3] = capsule['end_points'][1]
            self.viz.viewer[f'world/obstacle/sphere2{capsule["index"]}'].set_transform(T_Tmp@rot_mat)

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

    def moveCamera(self, cam_pos):
        self.viz.setCameraPosition(cam_pos)

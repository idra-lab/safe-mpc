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

np.random.seed(0)


### METHODS ###

def get_ocp(ocp_name, model) -> NaiveOCP:
    ocps = { 'naive': NaiveOCP,
             'zerovel': TerminalZeroVelOCP,
             'st': SoftTerminalOCP,
             'htwa': HardTerminalOCP,
             'receding': HardTerminalOCP,
             'real_receding': HardTerminalOCP,
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
                    'real_receding': HTWAController,
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
                    'real_receding': RealReceding }
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

def sphere_sphere_dist(obs,ee_expr):
    return (ee_expr - obs['position']).T @ (ee_expr - obs['position'])

def plane_sphere_dist(obs,ee_expr):
    return ee_expr[obs['perpendicular_axis']] - obs['bounds'][obs['real_bound']]

def randomize_model(urdf_file_path,noise_mass=None, noise_inertia=None, noise_cm_position=None):
    if (noise_mass!= None and noise_inertia!=None and noise_cm_position != None):
        inertia_fields = ['ixx','iyy','izz', 'ixy', 'iyz' , 'ixz']
        # Load the URDF file
        tree = ET.parse(urdf_file_path)
        root = tree.getroot()
        links = root.findall('link')
        for link in links:
            inertial = link.find('inertial')
            if inertial is not None:
                mass=inertial.find('mass')
                noise = float(mass.get('value')) * noise_mass / 100
                new_mass = float(mass.get('value')) + np.random.uniform(-noise, noise)
                mass.set('value', str(new_mass))

                #print(f'new_mass: {new_mass}')
                
                inertia = inertial.find('inertia')
                for i in inertia_fields:
                    noise =  abs(float(inertia.get(i))) * noise_inertia / 100
                    new_inertia = float(inertia.get(i))+np.random.uniform(-noise, noise)
                    inertia.set(i,str(new_inertia))

                #print(f'new_inertia: {new_inertia}')

                
                cm_pos = inertial.find('origin')
                pos = list(map(float, cm_pos.get('xyz').split(' ')))
                for e in pos:
                    noise = abs(e*noise_cm_position / 100)
                    #e += np.random.uniform(-noise, noise)
                cm_pos.set('xyz', ' '.join(map(str, pos)))

        # Write the modified URDF back to a file
        tree.write(urdf_file_path[:-5] + '_randomized.urdf', encoding='utf-8', xml_declaration=True)

def casadi_if_else(logic_var,expression,bounds):
        return(cs.if_else(logic_var > 0, 
                          expression, 
                          (bounds[0] + bounds[1])/2, True))


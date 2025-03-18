import casadi as cs
import numpy as np
from .utils import rot_mat_x,rot_mat_y,rot_mat_z
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel
import sympy as sym


# class to manage cost settings: until now only Q*error_ee*Q + R*u*R
class CostSettings():
    def __init__(self, model, traj = None):
        self.model = model

        if model.params.cost_type == 'LINEAR_LS':

        elif model.params.cost_type == 'NONLINEAR_LS':

        elif model.params.cost_type == 'EXTERNAL':
            t_glob = self.model.t_glob
            delta = t_glob - self.model.amodel.p[:3]    #self.model.ee_ref
            track_ee = delta.T @ (self.model.params.Q_weight * np.eye(3)) @ delta 
            self.ocp.model.cost_expr_ext_cost = track_ee + self.model.u.T @ (self.model.params.R_weight * np.eye(self.model.nu)) @ self.model.u if not(self.model.params.track_traj) else track_ee
            self.ocp.model.cost_expr_ext_cost_e = track_ee

        if model.params.track_traj == False:
            self.traj = np.tile(model.params.ee_ref, (model.params.n_steps))
        else:
            self.traj = traj

    def ee_cost(self):
        self

def generate_trajectory(params):
        if params.vel_const == False:
            velocity = 0
            acc = params.vel_max_traj/(params.n_steps*params.acc_time)
        else:
            velocity =  params.vel_max_traj
        traj = np.zeros((3,params.n_steps_tracking+1))
        theta = 0
        for i in range(traj.shape[1]):
            traj[:,i] = get_point_from_theta(theta) 
            theta = theta_next_from_vel(theta,velocity)
            if not(params.vel_const) and velocity <= params.vel_max_traj:
                velocity += acc * params.dt
        rot_mat__traj = rot_mat_x(params.theta_rot_traj[0])@rot_mat_y(params.theta_rot_traj[1])@rot_mat_z(params.theta_rot_traj[2])
        traj_to_track = rot_mat__traj @ traj + params.offset_traj
        return traj_to_track

def get_point_from_theta(theta,params):
    return np.array([(params.a_shape_8*np.cos(theta))/(1+np.sin(theta)**2),(params.a_shape_8*np.cos(theta)*np.sin(theta))/(1+np.sin(theta)**2),0] )

def theta_next_from_vel(theta,vel_set,params):
    a_sym,t_sym = sym.symbols('a t')
    x_sym = (a_sym*sym.cos(t_sym))/(1+sym.sin(t_sym)**2)
    y_sym = (a_sym*sym.cos(t_sym)*sym.sin(t_sym))/(1+sym.sin(t_sym)**2)

    dx=sym.simplify(sym.diff(x_sym,t_sym))
    dy=sym.simplify(sym.diff(y_sym,t_sym))

    dx_fun = sym.lambdify(t_sym,dx.subs({a_sym:params.dim_shape_8}))
    dy_fun = sym.lambdify(t_sym,dy.subs({a_sym:params.dim_shape_8})) 
    return theta+(vel_set/(np.sqrt(dx_fun(theta)**2+dy_fun(theta)**2)))*params.dt



params = Parameters('z1', rti=True)
model = AdamModel(params)

cost_settings = CostSettings('NONLINEAR_LS',cs.vertcat(model.t_glob,model.u),)



import casadi as cs
import numpy as np
from .utils import rot_mat_x,rot_mat_y,rot_mat_z
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel
import sympy as sym


class CostSettings():
    def __init__(self, cost_type: str, expr_0_N_minus_1, expr_N, sim_params, traj = None):
        self.cost_type = cost_type
        self.expr_0_N_minus_1 = expr_0_N_minus_1
        self.expr_N = expr_N
        
        if self.sim_params.track_traj == False:
            self.traj = np.tile(sim_params.ee_ref, (sim_params.n_steps))
        else:
            self.traj = traj

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



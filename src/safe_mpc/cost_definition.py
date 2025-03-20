import casadi as cs
import numpy as np
import sympy as sym
import scipy.linalg as lin


# class to manage cost settings: until now only Q*error_ee*Q + R*u*R
class AbstractCostSettings():
    def __init__(self, controller):
        self.controller = controller
        self.cost_type = None

    def set_ocp_cost_type(self):
        self.controller.ocp.cost.cost_type_0 = self.cost_type
        self.controller.ocp.cost.cost_type = self.cost_type
        self.controller.ocp.cost.cost_type_e = self.cost_type

class ReachTargetNLS(AbstractCostSettings):
    def __init__(self,controller):
        super().__init__(controller)
        self.cost_type = 'NONLINEAR_LS'
        self.set_ocp_cost_type()

        self.controller.ocp.model.cost_y_expr_0 = cs.vertcat(self.controller.model.t_glob - self.controller.ee_params,self.controller.model.u)
        self.controller.ocp.cost.W_0 = lin.block_diag(self.controller.model.params.Q_weight*np.eye(self.controller.model.t_glob.shape[0]),self.controller.model.params.R_weight*np.eye(self.controller.model.nu))
        self.controller.ocp.cost.yref_0 = np.hstack((np.zeros((self.controller.model.t_glob.shape[0],)), np.zeros(self.controller.model.nu)))

        self.controller.ocp.model.cost_y_expr = cs.vertcat(self.controller.model.t_glob - self.controller.ee_params,self.controller.model.u)
        self.controller.ocp.cost.W = lin.block_diag(self.controller.model.params.Q_weight*np.eye(self.controller.model.t_glob.shape[0]),self.controller.model.params.R_weight*np.eye(self.controller.model.nu))
        self.controller.ocp.cost.yref = np.hstack((np.zeros((self.controller.model.t_glob.shape[0],)), np.zeros(self.controller.model.nu)))

        self.controller.ocp.model.cost_y_expr_e = cs.vertcat(self.controller.model.t_glob - self.controller.ee_params)
        self.controller.ocp.cost.W_e = lin.block_diag(self.controller.model.params.Q_weight*np.eye(self.controller.model.t_glob.shape[0]))
        self.controller.ocp.cost.yref_e = np.zeros((self.controller.model.t_glob.shape[0],))

        self.traj = np.tile(self.controller.model.params.ee_ref, (self.controller.model.params.n_steps + 1 + self.controller.model.params.N, 1)).T

class ReachTargetEXT(AbstractCostSettings):
    def __init__(self, controller):
        super().__init__(controller)
        self.cost_type = 'EXTERNAL'
        self.set_ocp_cost_type()

        t_glob = self.controller.model.t_glob
        delta = t_glob - self.controller.ee_params    #self.controller.model.ee_ref
        track_ee = delta.T @ (self.controller.model.params.Q_weight * np.eye(3)) @ delta 
        self.controller.ocp.model.cost_expr_ext_cost = track_ee + self.controller.model.u.T @ (self.controller.model.params.R_weight * np.eye(self.controller.model.nu)) @ self.controller.model.u
        self.controller.ocp.model.cost_expr_ext_cost_e = track_ee

        self.traj = np.tile(self.controller.model.params.ee_ref, (self.controller.model.params.n_steps + 1 + self.controller.model.params.N, 1)).T

class Abstract8Tracking(AbstractCostSettings):
    def __init__(self,controller):
        super().__init__(controller)
        a_sym,t_sym = sym.symbols('a t')
        x_sym = (a_sym*sym.cos(t_sym))/(1+sym.sin(t_sym)**2)
        y_sym = (a_sym*sym.cos(t_sym)*sym.sin(t_sym))/(1+sym.sin(t_sym)**2)

        dx=sym.simplify(sym.diff(x_sym,t_sym))
        dy=sym.simplify(sym.diff(y_sym,t_sym))

        self.dx_fun = sym.lambdify(t_sym,dx.subs({a_sym:self.controller.model.params.dim_shape_8}))
        self.dy_fun = sym.lambdify(t_sym,dy.subs({a_sym:self.controller.model.params.dim_shape_8})) 
        self.generate_trajectory()

    def generate_trajectory(self):
        from .utils import rot_mat_x,rot_mat_y,rot_mat_z

        if self.controller.model.params.vel_const == False:
            velocity = 0
            acc = self.controller.model.params.vel_max_traj/(self.controller.model.params.n_steps*self.controller.model.params.acc_time)
        else:
            velocity =  self.controller.model.params.vel_max_traj
        traj = np.zeros((3,self.controller.model.params.n_steps_tracking+1+self.controller.model.params.N))
        theta = 0
        for i in range(traj.shape[1]):
            traj[:,i] = self.get_point_from_theta(theta) 
            theta = self.theta_next_from_vel(theta,velocity)
            if not(self.controller.model.params.vel_const) and velocity <= self.controller.model.params.vel_max_traj:
                velocity += acc 
        rot_mat__traj = rot_mat_x(self.controller.model.params.theta_rot_traj[0])@rot_mat_y(self.controller.model.params.theta_rot_traj[1])@rot_mat_z(self.controller.model.params.theta_rot_traj[2])
        self.traj = (rot_mat__traj[:3,:3] @ traj) + self.controller.model.params.offset_traj.reshape(3,1)

    def get_point_from_theta(self,theta):
        return np.array([(self.controller.model.params.dim_shape_8*np.cos(theta))/(1+np.sin(theta)**2),(self.controller.model.params.dim_shape_8*np.cos(theta)*np.sin(theta))/(1+np.sin(theta)**2),0] )

    def theta_next_from_vel(self,theta,vel_set):
        
        return theta+(vel_set/(np.sqrt(self.dx_fun(theta)**2+self.dy_fun(theta)**2)))*self.controller.model.params.dt


class Tracking8NLS(Abstract8Tracking):
    def __init__(self,controller):
        super().__init__(controller)
        self.cost_type = 'NONLINEAR_LS'
        self.set_ocp_cost_type()

        self.controller.ocp.model.cost_y_expr_0 = cs.vertcat(self.controller.model.t_glob - self.controller.ee_params)
        self.controller.ocp.cost.W_0 = lin.block_diag(self.controller.model.params.Q_weight*np.eye(self.controller.model.t_glob.shape[0]))
        self.controller.ocp.cost.yref_0 = np.zeros((self.controller.model.t_glob.shape[0],))

        self.controller.ocp.model.cost_y_expr = self.controller.ocp.model.cost_y_expr_0 
        self.controller.ocp.cost.W =  self.controller.ocp.cost.W_0
        self.controller.ocp.cost.yref = self.controller.ocp.cost.yref_0 

        self.controller.ocp.model.cost_y_expr_e = self.controller.ocp.model.cost_y_expr_0
        self.controller.ocp.cost.W_e = self.controller.ocp.cost.W_0
        self.controller.ocp.cost.yref_e = self.controller.ocp.cost.yref_0 

class Tracking8EXT(Abstract8Tracking):
    def __init__(self,controller):
        super().__init__(controller)
        self.cost_type = 'EXTERNAL'
        self.set_ocp_cost_type()

        t_glob = self.controller.model.t_glob
        delta = t_glob - self.controller.ee_params    #self.controller.model.ee_ref
        track_ee = delta.T @ (self.controller.model.params.Q_weight * np.eye(3)) @ delta 
        self.controller.ocp.model.cost_expr_ext_cost = track_ee
        self.controller.ocp.model.cost_expr_ext_cost_e = track_ee

# params = Parameters('z1', rti=True)
# model = AdamModel(params)



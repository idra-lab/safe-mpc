import casadi as cs
import numpy as np
import sympy as sym
import scipy.linalg as lin
import yaml

# class to manage cost settings: until now only Q*error_ee*Q + R*u*R
class AbstractCost():
    def __init__(self, model):
        self.model = model
        self.cost_type = None
        model.params.track_traj = False

    def set_ocp_cost_type(self,controller):
        #controller.ocp.cost.cost_type_0 = self.cost_type
        controller.ocp.cost.cost_type = self.cost_type
        controller.ocp.cost.cost_type_e = self.cost_type
    
    def set_cost_expr(self,controller):
        pass

    def set_solver_cost(self,controller):
        controller.cost = self
        self.set_ocp_cost_type(controller)
        self.set_cost_expr(controller)

    def get_reference_traj(self):
        return np.tile(self.model.ee_ref, (self.model.params.N + 1, 1)).T
    
    def update_trajectory(self):
        self.traj = np.tile(self.model.ee_ref, (self.model.params.n_steps + 1 + self.model.params.N, 1)).T

class ZeroCost(AbstractCost):
    def __init__(self,model):
        super().__init__(model)
        self.cost_type = 'LINEAR_LS'
        self.traj = np.tile(self.model.params.ee_ref, (self.model.params.n_steps + 1 + self.model.params.N, 1)).T
        # Linear LS cost (zero cost)

    def set_ocp_cost_type(self,controller):
        controller.ocp.cost.cost_type = self.cost_type
        controller.ocp.cost.cost_type_e = self.cost_type
    
    def set_cost_expr(self,controller):
        pass
        # self.Q = np.zeros((self.model.nx, self.model.nx))
        # self.R = np.zeros((self.model.nu, self.model.nu))

        # controller.ocp.cost.W = lin.block_diag(self.Q, self.R)
        # controller.ocp.cost.W_e = self.Q

        # controller.ocp.cost.Vx = np.zeros((self.model.ny, self.model.nx))
        # controller.ocp.cost.Vu = np.zeros((self.model.ny, self.model.nu))
        # controller.ocp.cost.Vx_e = np.zeros((self.model.nx, self.model.nx))

        # controller.ocp.cost.yref = np.zeros(self.model.ny)
        # controller.ocp.cost.yref_e = np.zeros(self.model.nx)


class ReachTargetNLS(AbstractCost):
    def __init__(self,model,Q,R):
        super().__init__(model)
        self.cost_type = 'NONLINEAR_LS'
        self.Q = Q
        self.R = R
        self.traj = np.tile(self.model.params.ee_ref, (self.model.params.n_steps + 1 + self.model.params.N, 1)).T

    def set_cost_expr(self,controller):
        controller.ocp.model.cost_y_expr_0 = cs.vertcat(controller.model.t_glob - controller.ee_params,controller.model.u)
        controller.ocp.cost.W_0 = lin.block_diag(self.Q*np.eye(controller.model.t_glob.shape[0]),
                                                      self.R*np.eye(controller.model.nu))
        controller.ocp.cost.yref_0 = np.zeros((controller.model.t_glob.shape[0] + controller.model.nu,))

        controller.ocp.model.cost_y_expr = controller.ocp.model.cost_y_expr_0
        controller.ocp.cost.W = controller.ocp.cost.W_0
        controller.ocp.cost.yref = controller.ocp.cost.yref_0

        controller.ocp.model.cost_y_expr_e = cs.vertcat(controller.model.t_glob - controller.ee_params)
        controller.ocp.cost.W_e = lin.block_diag(self.Q*np.eye(controller.model.t_glob.shape[0]))
        controller.ocp.cost.yref_e = np.zeros((controller.model.t_glob.shape[0],))

class ReachTargetEXT(AbstractCost):
    def __init__(self, model,Q,R):
        super().__init__(model)
        self.Q = Q
        self.R = R
        self.cost_type = 'EXTERNAL'
        self.traj = np.tile(model.params.ee_ref, (model.params.n_steps + 1 + model.params.N, 1)).T

    def set_cost_expr(self,controller):
        t_glob = controller.model.t_glob
        delta = t_glob - controller.ee_params    #controller.model.ee_ref
        track_ee = delta.T @ (self.Q * np.eye(3)) @ delta 
        controller.ocp.model.cost_expr_ext_cost = track_ee + controller.model.u.T @ (self.R * np.eye(controller.model.nu)) @ controller.model.u
        controller.ocp.model.cost_expr_ext_cost_e = track_ee
    
    def set_ocp_cost_type(self, controller):
        super().set_ocp_cost_type(controller)
        controller.ocp.solver_options.hessian_approx = "EXACT"

class AbstractTracking8():
    def __init__(self,model):
        self.model = model
        parameters_settings = yaml.load(open(model.params.ROOT_DIR + '/config.yaml'), Loader=yaml.FullLoader)

        model.params.n_steps=int(parameters_settings['n_steps_tracking'])
        model.params.n_steps_tracking=int(parameters_settings['n_steps_tracking'])
        model.params.dim_shape_8 = float(parameters_settings['dim_shape_8'])
        model.params.offset_traj = np.array(parameters_settings['offset_traj'])
        model.params.theta_rot_traj = np.array(parameters_settings['theta_rot_traj'])
        model.params.vel_max_traj = float(parameters_settings['vel_max_traj'])
        model.params.vel_const = bool(parameters_settings['vel_const'])
        model.params.acc_time = float(parameters_settings['acc_time'])

        self.traj = generate_8shape_trajectory(model.params)
        model.params.track_traj = True
        self.theta = 0
        self.vel = model.params.vel_max_traj if model.params.vel_const else 0
        self.acc = self.model.params.vel_max_traj/(self.model.params.n_steps*self.model.params.acc_time)


        a_sym,t_sym = sym.symbols('a t')
        x_sym = (a_sym*sym.cos(t_sym))/(1+sym.sin(t_sym)**2)
        y_sym = (a_sym*sym.cos(t_sym)*sym.sin(t_sym))/(1+sym.sin(t_sym)**2)
        dx=sym.simplify(sym.diff(x_sym,t_sym))
        dy=sym.simplify(sym.diff(y_sym,t_sym))

        self.x_fun = sym.lambdify(t_sym,x_sym.subs({a_sym:model.params.dim_shape_8}))
        self.y_fun = sym.lambdify(t_sym,y_sym.subs({a_sym:model.params.dim_shape_8}))
        self.dx_fun = sym.lambdify(t_sym,dx.subs({a_sym:model.params.dim_shape_8}))
        self.dy_fun = sym.lambdify(t_sym,dy.subs({a_sym:model.params.dim_shape_8}))

    def get_reference_trajectory(self):
        from .utils import rot_mat_x,rot_mat_y,rot_mat_z
        theta = self.theta
        vel = self.vel 
        self.theta += (vel/(np.sqrt(self.dx_fun(theta)**2+self.dy_fun(theta)**2)))*self.model.params.dt
        ref_traj = np.zeros(3,self.model.params.N + 1) 
        for i in range(self.model.params.N + 1):
            ref_traj[:,i] = [self.x_fun(theta),self.y_fun(theta),0]
            theta += (vel/(np.sqrt(self.dx_fun(theta)**2+self.dy_fun(theta)**2)))*self.model.params.dt
            if not(self.model.params.vel_const) and vel <= self.model.params.vel_max_traj:
                vel += self.acc 
        if not(self.model.params.vel_const) and self.vel <= self.model.params.vel_max_traj:
            self.vel += self.acc
        return ref_traj
    
    def update_trajectory(self):
        self.traj = generate_8shape_trajectory(self.model.params)

class Tracking8NLS(ReachTargetNLS, AbstractTracking8):
    def __init__(self,model,Q,R):
        super().__init__(model,Q,R)
        AbstractTracking8.__init__(self,model)
        self.cost_type = 'NONLINEAR_LS'

class Tracking8EXT(ReachTargetEXT):
    def __init__(self,model,Q,R):
        super().__init__(model,Q,R)
        AbstractTracking8.__init__(self,model)
        self.cost_type = 'EXTERNAL'

def generate_8shape_trajectory(params):
    from .utils import rot_mat_x,rot_mat_y,rot_mat_z

    a_sym,t_sym = sym.symbols('a t')
    x_sym = (a_sym*sym.cos(t_sym))/(1+sym.sin(t_sym)**2)
    y_sym = (a_sym*sym.cos(t_sym)*sym.sin(t_sym))/(1+sym.sin(t_sym)**2)

    dx=sym.simplify(sym.diff(x_sym,t_sym))
    dy=sym.simplify(sym.diff(y_sym,t_sym))

    dx_fun = sym.lambdify(t_sym,dx.subs({a_sym:params.dim_shape_8}))
    dy_fun = sym.lambdify(t_sym,dy.subs({a_sym:params.dim_shape_8}))

    if params.vel_const == False:
        velocity = 0
        acc = params.vel_max_traj/(params.n_steps*params.acc_time)
    else:
        velocity =  params.vel_max_traj
    traj = np.zeros((3,params.n_steps_tracking+1+params.N))
    theta = 0
    for i in range(traj.shape[1]):
        traj[:,i] = np.array([(params.dim_shape_8*np.cos(theta))/(1+np.sin(theta)**2),
                                (params.dim_shape_8*np.cos(theta)*np.sin(theta))/(1+np.sin(theta)**2)
                                ,0] )
        theta = theta+(velocity/(np.sqrt(dx_fun(theta)**2+dy_fun(theta)**2)))*params.dt
        if not(params.vel_const) and velocity <= params.vel_max_traj:
            velocity += acc 
    rot_mat__traj = rot_mat_x(params.theta_rot_traj[0])@rot_mat_y(params.theta_rot_traj[1])@rot_mat_z(params.theta_rot_traj[2])
    traj = (rot_mat__traj[:3,:3] @ traj) + params.offset_traj.reshape(3,1)
    return traj

class AbstractTrackingMovingCircle():
    def __init__(self,model):
        self.model = model

        parameters_settings = yaml.load(open(model.params.ROOT_DIR + '/config.yaml'), Loader=yaml.FullLoader)

        model.params.n_steps=int(parameters_settings['n_steps_tracking'])
        model.params.n_steps_tracking=int(parameters_settings['n_steps_tracking'])
        model.params.circle_rad = float(parameters_settings['circle_rad'])
        model.params.circle_offset_traj = np.array(parameters_settings['circle_offset_traj'])
        model.params.circle_traj_vel = float(parameters_settings['circle_traj_vel'])
        model.params.vel_const = bool(parameters_settings['vel_const'])
        model.params.circle_center_vel = float(parameters_settings['circle_center_vel'])
        model.params.acc_time = float(parameters_settings['acc_time'])

        model.params.track_traj = True

        self.theta = 0
        self.vel = model.params.circle_traj_vel if model.params.vel_const else 0
        self.acc = self.model.params.circle_traj_vel/(self.model.params.n_steps*self.model.params.acc_time)
        self.traj = generate_moving_circle_trajectory(self.model.params)

    def get_reference_trajectory(self):
        theta = self.theta
        vel = self.vel 
        self.theta += (self.vel/(np.sqrt(-np.sin(theta)**2 + np.cos(theta)**2)))*self.model.params.dt
        ref_traj = np.zeros(3,self.model.params.N + 1) 
        sign_vel = 1
        for i in range(self.model.params.N + 1):
            ref_traj[:,i] = self.model.params.circle_rad*np.array([-np.cos(theta),np.sin(theta),0]) + \
                            sign_vel*np.array([0,self.model.params.circle_center_vel * self.model.params.dt,0]) + \
                            np.array(self.model.params.circle_offset_traj)
            theta = theta+(vel/(np.sqrt(self.model.params.circle_rad*(np.sin(theta)**2 + np.cos(theta)**2))))*self.model.params.dt
            print(ref_traj[1,i])
            if sign_vel>0 and ref_traj[1,i] < -0.3:
                sign_vel=-1
            if sign_vel<0 and ref_traj[1,i] > 0.3:
                sign_vel=1
            if not(self.model.params.vel_const) and vel <= self.model.params.circle_traj_vel:
                vel += self.acc 
        if not(self.model.params.vel_const) and self.vel <= self.model.params.circle_traj_vel:
            self.vel += self.acc
        #ref_traj = ref_traj + self.model.params.circle_offset_traj.reshape(3,1) 
        return ref_traj

class TrackingMovingCircleNLS(ReachTargetNLS,AbstractTrackingMovingCircle):
    def __init__(self,model,Q,R):
        super().__init__(model,Q,R)
        AbstractTrackingMovingCircle.__init__(self,model)
        self.cost_type = 'NONLINEAR_LS'

    def update_trajectory(self):
        self.traj = generate_moving_circle_trajectory(self.model.params)

class TrackingMovingCircleEXT(ReachTargetEXT,AbstractTrackingMovingCircle):
    def __init__(self,model,Q,R):
        super().__init__(model,Q,R)
        AbstractTrackingMovingCircle.__init__(self,model)
        self.cost_type = 'EXTERNAL'

    def update_trajectory(self):
        self.traj = generate_moving_circle_trajectory(self.model.params)

def generate_moving_circle_trajectory(params):
    if params.vel_const == False:
        velocity = 0
        acc = params.circle_traj_vel/(params.n_steps*params.acc_time)
    else:
        velocity =  params.circle_traj_vel
    traj = np.zeros((3,params.n_steps_tracking+1+params.N))
    circle = np.zeros((3,params.n_steps_tracking+1+params.N))
    linear_mov = np.zeros((3,params.n_steps_tracking+1+params.N))
    theta = 0
    sign_vel = 1
    for i in range(traj.shape[1]):
        circle[:,i] = params.circle_rad*np.array([-np.cos(theta) ,np.sin(theta) ,0]) 
        linear_mov[:,i] = linear_mov[:,max(i-1,0)] - sign_vel*np.array([0,params.circle_center_vel * params.dt,0]) 
        traj[:,i] = circle[:,i] + linear_mov[:,i] + np.array(params.circle_offset_traj)
        theta = theta+(velocity/(np.sqrt(params.circle_rad*(np.sin(theta)**2 + np.cos(theta)**2))))*params.dt
        if sign_vel>0 and traj[1,i] < -0.5:
                sign_vel=-1
        if sign_vel<0 and traj[1,i] > 0.5:
            sign_vel=1
        if not(params.vel_const) and velocity <= params.circle_traj_vel:
            velocity += acc 
    #traj = traj + params.circle_offset_traj.reshape(3,1) 

    return traj    





# params = Parameters('z1', rti=True)
# model = AdamModel(params)



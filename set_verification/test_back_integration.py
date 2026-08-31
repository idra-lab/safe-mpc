import numpy as np

def f(x):
    r = 2.0
    return np.exp( x)

def rk4_step(x, dt):
    k1 = f(x)
    k2 = f(x + dt/2 * k1)
    k3 = f(x + dt/2 * k2)
    k4 = f(x + dt  * k3)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
def rk4_back_step(x, dt):
    k1 = -f(x)
    k2 = -f(x + dt/2 * k1)
    k3 = -f(x + dt/2 * k2)
    k4 = -f(x + dt  * k3)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
# example
x = 0.1
dt = 0.1
print(f"t={( 0)*dt:.1f}  x={x:.6f}")
for i in range(5):
    x = rk4_step(x, dt)
    print(f"t={( i+1)*dt:.1f}  x={x:.6f}")
for i in range(5):
    x = rk4_back_step(x, dt)
    print(f"t={( i+1+5)*dt:.1f}  x={x:.6f}")

def f(x,u,dt):
    dim = x.shape[0]
    res = np.zeros(dim)
    res[:dim//2] = x[:dim//2] + dt * x[dim//2:] + 0.5 * dt**2 * u
    res[dim//2:] = x[dim//2:] + dt * u
    return res

def f_back(x,u,dt):
    dim = x.shape[0]
    res = np.zeros(dim)
    res[:dim//2] = x[:dim//2] - dt * x[dim//2:] + 0.5 * dt**2 * u
    res[dim//2:] = x[dim//2:] - dt * u
    return res


x = np.random.rand(4) * 100

n_steps = 20
dt = 0.005  

u = np.random.rand(n_steps//2, x.shape[0]//2) * 10
print(x)
for i in range(n_steps//2):
    x = f(x, u[i], dt)
    print(f"Step {i+1}, x: {x}, u: {u[i]}")
for i in range(n_steps//2):
    x = f_back(x, u[n_steps//2 - 1 - i], dt)
    print(f"Back Step {i+1}, x: {x}, u: {u[n_steps//2 - 1 - i]}")
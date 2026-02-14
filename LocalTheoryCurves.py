"""
Code demonstration for the fundamental theorem of the local  theory of curves
(In R^3 (space curves))
"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid
from mpl_toolkits.mplot3d import Axes3D
import sympy.vector
from scipy.interpolate import interp1d
from scipy.special import ellipe

def system_of_eqns(s, y, kappa, tau):
    # q0 = point
    # T = tangent vector, N = normal, B = binormal
    # these are going to be arrays or matrices

    q0 = y[0:3]
    T = y[3:6]
    N = y[6:9]
    B = y[9:12]
    
    kappa = kappa(s)
    tau = tau(s)
    
    dr_ds = T
    dT_ds = kappa*N
    dN_ds = -kappa*T + tau*B
    dB_ds = -tau*N
    return np.concatenate([dr_ds, dT_ds, dN_ds, dB_ds])

def plot_frenet(q0, kappa, tau, s_range, initial_T = None, initial_N = None, plane_curve=False):
    if initial_T is None:
        initial_T = np.array([1.0, 0.0, 0.0])
    if initial_N is None:
        initial_N = np.array([0.0, 1.0, 0.0])

    T0 = initial_T/np.linalg.norm(initial_T)
    N0 = initial_N - np.dot(initial_N, T0)*T0 # gram schmidt orthog process, proj is simplified because <T0, T0>=1
    N0 = N0/np.linalg.norm(N0)
    B0 = np.cross(T0, N0)

    y0 = np.concatenate([q0, T0, N0, B0])

    sol = solve_ivp(
            lambda s, y: system_of_eqns(s, y, kappa, tau),
            s_range,
            y0,
            dense_output=True, max_step=0.01
    )

    s_plot = np.linspace(s_range[0], s_range[1], 1000)
    y_plot = sol.sol(s_plot)

    x = y_plot[0, :]
    y = y_plot[1, :]
    z = y_plot[2, :]

    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot(x, y, z, 'b-', linewidth=2)
    ax1.scatter(*q0, color='red', s=100, label='q0')
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title('Local Theory of Curves in R^3')
    ax1.legend()
    ax1.grid(True)

    plt.tight_layout()
    plt.show()

    return x, y, z, s_plot

# circular helix example
def circular_helix(a, b, q0):
    kappa = lambda s: a/(a**2 + b**2)
    tau = lambda s: b/(a**2 + b**2)
    
    x, y, z, s = plot_frenet(q0=q0, kappa=kappa, tau=tau, s_range=[0,20])

def curvature(r, sym):
    # for a curve in R^3: kappa = ||r' x r''||/||r'||^3
    # r is going to have a symbolic variable sym
    r_prime = sp.Matrix(sp.diff(r, sym))
    r_prime2 = sp.Matrix(sp.diff(r_prime, sym))
    numerator_cross = r_prime.cross(r_prime2)
    # get the magnitude
    numerator = numerator_cross.norm()
    r_prime_mag = r_prime.norm()
    denom = r_prime_mag**3
    kappa = numerator/denom
    return kappa

def torsion(r, sym):
    r_prime = sp.Matrix(sp.diff(r, sym))
    r_prime2 = sp.Matrix(sp.diff(r_prime, sym))
    r_prime3 = sp.Matrix(sp.diff(r_prime2, sym))
    det_mat = sp.Matrix.hstack(r_prime, r_prime2, r_prime3)
    numerator = det_mat.det()
    r_cross = r_prime.cross(r_prime2)
    r_cross_mag = r_cross.norm()
    denom = r_cross_mag**2
    tau = numerator/denom
    return tau

def t_to_arclength(t_val, r, sym):
    # t to s
    r_prime = sp.Matrix(sp.diff(r, sym))
    s = r_prime.norm()
    s_numeric = sp.lambdify(sym, s, 'numpy')
    return s_numeric(t_val)
    # return_array = np.zeros_like(t_val)
    # for i, val in zip(range(len(t_val)), t_val):
    #     s_val = float(s.subs(sym, val).evalf())
    #     return_array[i] = s_val
    # return return_array

"""
Note: one slightly annoying aspect of twisted_cubic and other functions here is that I am using
the parametrization r of the curve in order to compute my kappa and tau functions.
So the local theory of curves still holds because the actual plot is only being generated
from a point on the curve (q0), kappa, and tau, but I had to computationally get kappa and tau
using the parametrization.
"""
def twisted_cubic(t_min, t_max, t_value):
    t = sp.Symbol('t')
    r = sp.Array([t, t**2, t**3])
    r_prime = sp.Matrix(sp.diff(r, t))
    r_2prime = sp.Matrix(sp.diff(r_prime, t))
    
    t_values = np.linspace(t_min, t_max, 1000)
    s_values = np.zeros_like(t_values)
    s_values[1:] = cumulative_trapezoid(t_to_arclength(t_values, r, t), t_values)
    t_func = interp1d(s_values, t_values, kind='cubic', fill_value='extrapolate')
    
    kappa_sym = curvature(r, t)
    tau_sym = torsion(r, t)

    kappa_numeric = sp.lambdify(t, kappa_sym, 'numpy')
    tau_numeric = sp.lambdify(t, tau_sym, 'numpy')

    def kappa_func(s):
        t_value = t_func(s) 
        kappa_val = float(kappa_numeric(t_value))
        return kappa_val
    def tau_func(s):
        t_value = t_func(s)
        tau_val = float(tau_numeric(t_value))
        return tau_val

    q0 = r.subs(t, t_value)
    q0 = np.array(q0, dtype=float)
    # q0 = t_func(q0)
    print(q0)

    # evaluate r' at t_value
    T0 = r_prime.subs(t, t_value)
    N0 = r_2prime.subs(t, t_value)
    T0 = np.array(T0, dtype=float).flatten()
    N0 = np.array(N0, dtype=float).flatten()
    print('T0 = ', T0)
    print("N0 = ", N0)


    x, y, z, s = plot_frenet(q0=q0, kappa=kappa_func, tau=tau_func, s_range=[s_values[0], s_values[-1]], initial_T=T0, initial_N=N0)
    
def semicubical_parabola(t_min, t_max, q0=[0,0, 0]):
    # TODO: in progress, doesn't work because needs to be plotted on a plane not 3d
    # c(t) = (t**2, t***3): reference when choosing a point q0
    t = sp.Symbol('t')
    r = sp.Array([t**2, t**3, 0])
    r_prime = sp.Matrix(sp.diff(r, t)) # tangent vector
    r_2prime = sp.Matrix(sp.diff(r_prime, t)) # normal vector
   
    eps = 0.3
    if t_min < eps and t_max > eps:
        t_min = eps

    t_values = np.linspace(t_min, t_max, 1000)
    t_values = t_values[np.abs(t_values) > eps]
    s_values = np.zeros_like(t_values)
    s_values[1:] = cumulative_trapezoid(t_to_arclength(t_values, r, t), t_values)
    t_func = interp1d(s_values, t_values, kind='cubic', fill_value='extrapolate')

    kappa_sym = curvature(r, t)
    tau_sym = sp.Integer(0)

    kappa_numeric = sp.lambdify(t, kappa_sym, 'numpy')
    tau_numeric = sp.lambdify(t, tau_sym, 'numpy')
    
    def kappa_func(s):
        t_value = t_func(s) 
        kappa_val = float(kappa_numeric(t_value))
        return kappa_val
    def tau_func(s):
        t_value = t_func(s)
        tau_val = float(tau_numeric(t_value))
        return tau_val
    
    t_start = t_values[0]
    q0 = r.subs(t, t_start)
    q0 = np.array(q0, dtype=float)

    T0 = r_prime.subs(t, t_start)
    N0 = r_2prime.subs(t, t_start)
    T0 = np.array(T0, dtype=float).flatten()
    N0 = np.array(N0, dtype=float).flatten()
    print("q0 = ", q0)
    print('T0 = ', T0)
    print("N0 = ", N0)

    x, y, z, s = plot_frenet(q0=q0, kappa=kappa_func, tau=tau_func, s_range=[s_values[0], s_values[-1]], initial_T=T0, initial_N=N0)


def catenary():
    pass


def viviani_curve(t_min, t_max, a, q0=[2,0,0]):
    # TODO: not quite right yet...
    # a = cylinder raidus
    # also q0 that works: 0, 0, +-2 for a=1
    def get_s(a, t):
        s = 2*a*np.sqrt(2)*ellipe(0.5*t)
        return s

    t_range = np.linspace(t_min, t_max, 1000)
    s_range = get_s(a, t_range)

    kappa_func = lambda t: (np.sqrt(13 + 3*np.cos(t)))/(a*(3 + np.cos(t))**1.5)
    tau_func = lambda t: (6*np.cos(0.5*t))/(a*(13 + 3*np.cos(t)))

    x, y, z, s = plot_frenet(q0=q0, kappa=kappa_func, tau=tau_func, s_range=[t_min, t_max])

def conical_spiral():
    pass




# twisted_cubic(-2, 2, 0)
viviani_curve(-10, 10, 1) 

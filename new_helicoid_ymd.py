import casadi as ca
import numpy as np
import math
import matplotlib.pyplot as plt
import time

ref_load = 1500
pCx1 = 1.532
pDx1 = 2.0217
pDx2 = -1.3356e-12
pEx1 = -0.53967
pKx1 = 31.5328
pKx3 = -0.83511
lambda_mux = 1

pCy1 = 1.5
pDy1 = 2.3298
pDy2 = -0.5
pEy1 = -0.052474
pKy1 = -42.8074
pKy2 = 1.7679
lambda_muy = 1

# constants
m = 262
Iz = 130
wheelbase = 1.53
b = 0.72675
a = wheelbase - b
h = 0.24
rho_air = 1.2
Cd = 1.1 # drag coef
Cs = 0.0
Cl = 2.5 # downforce coef
A = 1.1
alpha_max = np.deg2rad(20)
kappa_max = 0.5
P_max = 80 # kW
CLfA = Cl*A
CLrA = Cl*A

vehicleMass = m
trackwidth = 1.2
roll_stiffness = 0.53

"""
Functions for computation
"""
def mf_fx_fy(kappa, alpha, Fz):
    global ref_load

    error_eps = 1e-6
    # calculate the coefs
    dfz = (Fz - ref_load)/ref_load
    Kx = Fz*pKx1*ca.exp(pKx3*dfz)
    Ex = pEx1
    Dx = (pDx1 + pDx2*dfz)*lambda_mux
    Cx = pCx1
    Bx = Kx/(Cx*Dx*Fz)
    
    Ky = ref_load*pKy1*ca.sin(2*ca.atan(Fz/(pKy2*ref_load)))
    Ey = pEy1
    Dy = (pDy1 + pDy2*dfz)*lambda_muy
    Cy = pCy1
    By = Ky/(Cy*Dy*Fz)

    # magic formula
    sig_x = kappa/(1 + kappa)
    sig_y = alpha/(1 + kappa)
    sig = ca.sqrt((sig_x**2) + (sig_y**2))

    Fx = Fz*(sig_x/(sig + error_eps))*Dx*ca.sin(Cx * ca.atan(Bx*sig - Ex*(Bx*sig - ca.atan(Bx*sig))))
    Fy = Fz*(sig_y/(sig + error_eps))*Dy*ca.sin(Cy*ca.atan(By*sig - Ey*(By*sig - ca.atan(By*sig))))


    return Fx, Fy


def normal_loads(ax, ay, u):
    global CLfA, CLrA, rho_air, vehicleMass, a, b, trackwidth, h, roll_stiffness
    FLf = 0.5*CLfA*rho_air*(u**2)
    FLr = 0.5*CLrA*rho_air*(u**2)

    Nfl = 0.5*vehicleMass*g*(b/(a+b)) - 0.5*vehicleMass*ax*(h/(a+b)) + vehicleMass*ay*(h/trackwidth)*roll_stiffness + 0.5*FLf        
    Nfr = 0.5*vehicleMass*g*(b/(a+b)) - 0.5*vehicleMass*ax*(h/(a+b)) - vehicleMass*ay*(h/trackwidth)*roll_stiffness + 0.5*FLf        
    Nrl = 0.5*vehicleMass*g*(a/(a+b)) + 0.5*vehicleMass*ax*(h/(a+b)) + vehicleMass*ay*(h/trackwidth)*(1 - roll_stiffness) + 0.5*FLr        
    Nrr = 0.5*vehicleMass*g*(a/(a+b)) + 0.5*vehicleMass*ax*(h/(a+b)) - vehicleMass*ay*(h/trackwidth)*(1 - roll_stiffness) + 0.5*FLr

    Nfl = ca.fmax(Nfl, 1e-3)
    Nfr = ca.fmax(Nfr, 1e-3)
    Nrl = ca.fmax(Nrl, 1e-3)
    Nrr = ca.fmax(Nrr, 1e-3)


    return Nfl, Nfr, Nrl, Nrr

def calc_gb(rho0, phi, k, beta):
    gb_coef = g/(np.sqrt(((rho0)**2) * np.cos(phi)**2 + k**2)) 
    gb_x = gb_coef*(-0.5*rho0*np.sin(beta)*np.sin(2*phi) - k*np.cos(beta))
    gb_y = gb_coef*(0.5*rho0*np.cos(beta)*np.sin(2*phi) - k*np.sin(beta))
    gb_z = gb_coef*(rho0*np.cos(phi)**2)

    return gb_x, gb_y, gb_z


def calc_ab(V, Ay, rho0, phi, k, beta):
    ab_coef = ((V**2)*rho0*np.cos(phi))/(np.sqrt((rho0**2)*np.cos(phi)**2 + k**2))
    # ab_coef = Ay
    ab_x = ab_coef*(np.cos(phi)*np.sin(beta))
    ab_y = ab_coef*(-np.cos(phi)*np.cos(beta))
    ab_z = ab_coef*(np.sin(phi))

    return ab_x, ab_y, ab_z

def forces(AxIn, AyIn, beta, delta, kappa):
    global V, m, a, b, h, g, rho_air, A, Cd, Cs, Cl, kappa, trackwidth
    global rho0, phi, k

    R = (V**2)/(Ay_i)
    r = V/R
    
    # aero forces 
    fx_aero = 0.5*rho_air*A*Cd*(V**2) # drag
    fy_aero = 0.5*rho_air*A*Cs*(V**2)
    fz_aero = 0.5*rho_air*A*Cl*(V**2) # downforce

    # normal loads
    Nfl, Nfr, Nrl, Nrr = normal_loads(AxIn, AyIn, V)

    # slip angles
    alpha_r = np.atan2((b/R) - np.sin(beta), np.cos(beta))
    alpha_f = delta - np.atan2(np.sin(beta) + (a/R), np.cos(beta))

    # lat and long tire forces
    fx_fl, fy_fl = mf_fx_fy(kappa[0], alpha_f, Nfl)
    fx_fr, fy_fr = mf_fx_fy(kappa[1], alpha_f, Nfr)
    fx_rl, fy_rl = mf_fx_fy(kappa[2], alpha_r, Nrl)
    fx_rr, fy_rr = mf_fx_fy(kappa[3], alpha_r, Nrr)

    # fy_r = fy_rl + fy_rr
    # fx_r = fy_rl*np.sin(rearToe) + fy_rr*np.sin(rearToe)

    # fy_f = fy_fr*np.cos(delta - frontToe) + fy_fl*np.cos(delta + frontToe)
    # fx_f = -fy_fl*np.sin(frontToe + delta) + fy_fr*np.sin(frontToe - delta)

    # fy = fy_f + fy_r
    # fx = fx_r + fx_f - fx_aero 

    # fx_left = -fy_fl*np.sin(frontToe + delta) - fy_rl*np.sin(rearToe)
    # fx_right = fy_fr*np.sin(frontToe - delta) + fy_rr*np.sin(rearToe)
    

    ax_b, ay_b, az_b = calc_ab(V, AyIn, rho0, phi, k, beta)
    gx_b, gy_b, gz_b = calc_gb(rho0, phi, k, beta)


    Fx_b = np.cos(delta)*fx_f - np.sin(delta)*fy_f + fx_r - fx_aero + m*gx_b
    Fy_b = np.cos(delta)*fy_f + ca.sin(delta)*fx_f + fy_r + fy_aero + m*gy_b

    Mz = a*(fy_f*np.cos(delta) + fx_f*np.sin(delta)) - b*fy_r
    # Y = fy_f*a - fy_r*b + (fx_left + fx_right)*trackwidth - Mz


    return



# MAIN IMPLEMENTATION
AxIn = 0
V = 10
V_dot = 0
phi = np.deg2rad(0)
k = 0 # m/rad
g = 9.807

downforce = 0.5*Cl*A*(V**2)
drag = 0.5*Cd*A*(V**2)
maxAlpha = np.deg2rad(20)
kappa = [0,0,0,0] # FL FR RL RR
maxDelta = np.deg2rad(11)
maxBeta = np.deg2rad(7)

# surface params - right helicoid
rho0 = 0.5
phi = 0
k = -1

i = 1
steps = 19
sweep = 40
tolerance = 0.0001
k = 0
tireFzMin = 1000

delta_range = np.linspace(-deltaMax, deltaMax, steps)
beta_range = np.linspace(-betaMax, betaMax, sweep)

graph_ay = []
graph_yaw = []
graph_num = []
graph_beta = []
graph_delta = []

for delta in delta_range:
    k = 1
    i = 1
    for beta in beta_range:
        AyIn = 2
        Ay = 1

        while abs(AyIn - Ay) > tolerance:
            AyInPrev = AyIn
            AyIn = 0.7*Ay + 0.3*AyInPrev

            # FORCES
             

        graph_ay.append(Ay)
        graph_yaw.append(Y)
        graph_num.append(2)
        graph_beta.append(beta)
        graph_delta.append(delta)

for beta in beta_range:
    k = k+1
    i = 1
    for delta in delta_range:
        AyIn = 2
        Ay = 0
        while abs(AyIn - Ay) > tolerance:
            AyInPrev = AyIn
            AyIn = 0.7*Ay + 0.3*AyInPrev

            # FORCES
            pass

        graph_ay.append(Ay)
        graph_yaw.append(Y)
        graph_num.append(2)
        graph_beta.append(beta)
        graph_delta.append(delta)


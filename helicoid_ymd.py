# generate a YMD for a fsae car going around a right helicoid
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
Cd = 1.1
Cs = 0.0
Cl = 0.15
A = 1.1
alpha_max = np.deg2rad(20)
kappa_max = 0.5
P_max = 80 # kW
CLfA = Cl*A
CLrA = Cl*A

vehicleMass = m
trackwidth = 1.2
roll_stiffness = 0.53

V = 10
V_dot = 0
phi = np.deg2rad(0)
k = 0 # m/rad

g = 9.807

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

def sols_mf_fx_fy(kappa, alpha, Fz):
    global ref_load

    error_eps = 1e-6
    # calculate the coefs
    dfz = (Fz - ref_load)/ref_load
    Kx = Fz*pKx1*np.exp(pKx3*dfz)
    Ex = pEx1
    Dx = (pDx1 + pDx2*dfz)*lambda_mux
    Cx = pCx1
    Bx = Kx/(Cx*Dx*Fz)
    
    Ky = ref_load*pKy1*np.sin(2*np.atan(Fz/(pKy2*ref_load)))
    Ey = pEy1
    Dy = (pDy1 + pDy2*dfz)*lambda_muy
    Cy = pCy1
    By = Ky/(Cy*Dy*Fz)

    # magic formula
    sig_x = kappa/(1 + kappa)
    sig_y = alpha/(1 + kappa)
    sig = np.sqrt((sig_x**2) + (sig_y**2))

    Fx = Fz*(sig_x/(sig + error_eps))*Dx*np.sin(Cx * np.atan(Bx*sig - Ex*(Bx*sig - np.atan(Bx*sig))))
    Fy = Fz*(sig_y/(sig + error_eps))*Dy*np.sin(Cy*np.atan(By*sig - Ey*(By*sig - np.atan(By*sig))))


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

def sols_normal_loads(ax, ay, u):
    global CLfA, CLrA, rho_air, vehicleMass, a, b, trackwidth, h, roll_stiffness
    FLf = 0.5*CLfA*rho_air*(u**2)
    FLr = 0.5*CLrA*rho_air*(u**2)

    Nfl = 0.5*vehicleMass*g*(b/(a+b)) - 0.5*vehicleMass*ax*(h/(a+b)) + vehicleMass*ay*(h/trackwidth)*roll_stiffness + 0.5*FLf        
    Nfr = 0.5*vehicleMass*g*(b/(a+b)) - 0.5*vehicleMass*ax*(h/(a+b)) - vehicleMass*ay*(h/trackwidth)*roll_stiffness + 0.5*FLf        
    Nrl = 0.5*vehicleMass*g*(a/(a+b)) + 0.5*vehicleMass*ax*(h/(a+b)) + vehicleMass*ay*(h/trackwidth)*(1 - roll_stiffness) + 0.5*FLr        
    Nrr = 0.5*vehicleMass*g*(a/(a+b)) + 0.5*vehicleMass*ax*(h/(a+b)) - vehicleMass*ay*(h/trackwidth)*(1 - roll_stiffness) + 0.5*FLr

    Nfl = max(Nfl, 1e-3)
    Nfr = max(Nfr, 1e-3)
    Nrl = max(Nrl, 1e-3)
    Nrr = max(Nrr, 1e-3)


    return Nfl, Nfr, Nrl, Nrr

def curvature_torsion(rho0, phi, k):
    kappa = (rho0*np.cos(phi))/(np.sqrt((rho0)**2 * np.cos(phi)**2 + k**2))
    tau = (-k)/(np.sqrt(((rho0)**2) * np.cos(phi)**2 + k**2))
    return kappa, tau

def calc_gb(rho0, phi, k, beta):
    gb_coef = g/(ca.sqrt(((rho0)**2) * ca.cos(phi)**2 + k**2)) 
    gb_x = gb_coef*(-0.5*rho0*ca.sin(beta)*ca.sin(2*phi) - k*ca.cos(beta))
    gb_y = gb_coef*(0.5*rho0*ca.cos(beta)*ca.sin(2*phi) - k*ca.sin(beta))
    gb_z = gb_coef*(rho0*ca.cos(phi)**2)

    return gb_x, gb_y, gb_z

def sols_calc_gb(rho0, phi, k, beta):
    gb_coef = g/(np.sqrt(((rho0)**2) * np.cos(phi)**2 + k**2)) 
    gb_x = gb_coef*(-0.5*rho0*np.sin(beta)*np.sin(2*phi) - k*np.cos(beta))
    gb_y = gb_coef*(0.5*rho0*np.cos(beta)*np.sin(2*phi) - k*np.sin(beta))
    gb_z = gb_coef*(rho0*np.cos(phi)**2)

    return gb_x, gb_y, gb_z

def calc_ab(V, Ay, rho0, phi, k, beta):
    # ab_coef = ((V**2)*rho0*ca.cos(phi))/(ca.sqrt((rho0**2)*ca.cos(phi)**2 + k**2))
    ab_coef = Ay
    ab_x = ab_coef*(ca.cos(phi)*ca.sin(beta))
    ab_y = ab_coef*(-ca.cos(phi)*ca.cos(beta))
    ab_z = ab_coef*(ca.sin(phi))

    return ab_x, ab_y, ab_z

def sols_calc_ab(V, Ay, rho0, phi, k, beta):
    # ab_coef = ((V**2)*rho0*np.cos(phi))/(np.sqrt((rho0**2)*np.cos(phi)**2 + k**2))
    ab_coef = Ay
    ab_x = ab_coef*(np.cos(phi)*np.sin(beta))
    ab_y = ab_coef*(-np.cos(phi)*np.cos(beta))
    ab_z = ab_coef*(np.sin(phi))
    
    return ab_x, ab_y, ab_z

def force_calcs(V, rho0, phi, k, ax, Ay_i, beta_i, kf_i, kr_i, delta):
    global m, a, b, h, g, rho_air, P_max, A, Cd, Cs, Cl
    Ay_i = ca.if_else(ca.fabs(Ay_i) > 0.1, Ay_i, ca.sign(Ay_i)*0.1)
    R = (V**2)/(Ay_i)
    r = V/R

    # normal loads
    Nfl, Nfr, Nrl, Nrr = normal_loads(ax, Ay_i, V)

    # slip angles
    alpha_r = ca.atan2((b/R) - ca.sin(beta_i), ca.cos(beta_i))
    alpha_f = delta - ca.atan2(ca.sin(beta_i) + (a/R), ca.cos(beta_i))

    # lateral and longitudinal tire forces
    fx_fl, fy_fl = mf_fx_fy(kf_i, alpha_f, Nfl)
    fx_fr, fy_fr = mf_fx_fy(kf_i, alpha_f, Nfr)
    fx_rl, fy_rl = mf_fx_fy(kr_i, alpha_r, Nrl)
    fx_rr, fy_rr = mf_fx_fy(kr_i, alpha_r, Nrr)
    
    fx_f = fx_fl + fx_fr
    fy_f = fy_fl + fy_fr

    fx_r = fx_rl + fx_rr
    fy_r = fy_rl + fy_rr

    ax_b, ay_b, az_b = calc_ab(V, Ay_i, rho0, phi, k, beta_i)
    gx_b, gy_b, gz_b = calc_gb(rho0, phi, k, beta_i)


    # aero forces
    fx_aero = 0.5*rho_air*A*Cd*(V**2)
    fy_aero = 0.5*rho_air*A*Cs*(V**2)
    fz_aero = 0.5*rho_air*A*Cl*(V**2)

    Fx_b = ca.cos(delta)*fx_f - ca.sin(delta)*fy_f + fx_r - fx_aero + m*gx_b
    Fy_b = ca.cos(delta)*fy_f + ca.sin(delta)*fx_f + fy_r + fy_aero + m*gy_b

    forces = {'R': R, 'fz_fl': Nfl, 'fz_fr': Nfr, 'fz_rl': Nrl, 'fz_rr': Nrr, 'alpha_r': alpha_r, 'alpha_f': alpha_f, 'fx_f': fx_f, 'fy_f': fy_f,
              'fx_r': fx_r, 'fy_r': fy_r, 'ax_b': ax_b, 'ay_b': ay_b, 'az_b': az_b, 'gx_b': gx_b, 'gy_b': gy_b, 'gz_b': gz_b,
              'fx_aero': fx_aero, 'fy_aero': fy_aero, 'fz_aero': fz_aero, 'Fx_b': Fx_b, 'Fy_b': Fy_b}
    return forces

def sols_force_calcs(V, rho0, phi, k, ax, Ay_i, beta_i, kf_i, kr_i, delta):
    global m, a, b, h, g, rho_air, P_max, A, Cd, Cs, Cl
    if abs(Ay_i) > 0.1:
        R = (V**2)/(Ay_i)
    else:
        R = (V**2)/(np.sign(Ay_i)*0.1)
    r = V/R


    # normal loads
    Nfl, Nfr, Nrl, Nrr = sols_normal_loads(ax, Ay_i, V)

    # slip angles
    alpha_r = np.atan2((b/R) - np.sin(beta_i), np.cos(beta_i))
    alpha_f = delta - np.atan2(np.sin(beta_i) + (a/R), np.cos(beta_i))

    # lateral and longitudinal tire forces
    fx_fl, fy_fl = sols_mf_fx_fy(kf_i, alpha_f, Nfl)
    fx_fr, fy_fr = sols_mf_fx_fy(kf_i, alpha_f, Nfr)
    fx_rl, fy_rl = sols_mf_fx_fy(kr_i, alpha_r, Nrl)
    fx_rr, fy_rr = sols_mf_fx_fy(kr_i, alpha_r, Nrr)
    
    fx_f = fx_fl + fx_fr
    fy_f = fy_fl + fy_fr

    fx_r = fx_rl + fx_rr
    fy_r = fy_rl + fy_rr


    ax_b, ay_b, az_b = sols_calc_ab(V, Ay_i, rho0, phi, k, beta_i)
    gx_b, gy_b, gz_b = sols_calc_gb(rho0, phi, k, beta_i)

    # aero forces
    fx_aero = 0.5*rho_air*A*Cd*(V**2)
    fy_aero = 0.5*rho_air*A*Cs*(V**2)
    fz_aero = 0.5*rho_air*A*Cl*(V**2)

    Fx_b = np.cos(delta)*fx_f - np.sin(delta)*fy_f + fx_r - fx_aero + m*gx_b
    Fy_b = np.cos(delta)*fy_f + np.sin(delta)*fx_f + fy_r + fy_aero + m*gy_b

    forces = {'R': R, 'fz_fl': Nfl, 'fz_fr': Nfr, 'fz_rl': Nrl, 'fz_rr': Nrr, 'alpha_r': alpha_r, 'alpha_f': alpha_f, 'fx_f': fx_f, 'fy_f': fy_f,
              'fx_r': fx_r, 'fy_r': fy_r, 'ax_b': ax_b, 'ay_b': ay_b, 'az_b': az_b, 'gx_b': gx_b, 'gy_b': gy_b, 'gz_b': gz_b,
              'fx_aero': fx_aero, 'fy_aero': fy_aero, 'fz_aero': fz_aero, 'Fx_b': Fx_b, 'Fy_b': Fy_b}
    return forces




def ocp(delta, V, V_dot, ay_min, ay_max, beta_guess, kf_guess, kr_guess, N=50):
    global m, a, b, h, g, rho_air, P_max, A, Cd, Cs, Cl
    global rho0, phi, k
    global alpha_max, kappa_max
    
    ax = V_dot
    
    opti = ca.Opti()

    # states
    beta = opti.variable(N)
    kappa_f = opti.variable(N)
    kappa_r = opti.variable(N)

    # controls
    beta_prime = opti.variable(N-1)
    kappa_f_prime = opti.variable(N-1)
    kappa_r_prime = opti.variable(N-1)

    # Ay = opti.parameter(N)
    ay_range = np.linspace(ay_min, ay_max, N)

    J = 0
    for i in range(N-1):
        beta_i = beta[i]
        kf_i = kappa_f[i]
        kr_i = kappa_r[i]
        Ay_i = ay_range[i] # TODO: correct?
        Ay_i = ca.if_else(ca.fabs(Ay_i) > 0.1, Ay_i, ca.sign(Ay_i)*0.1)

        forces = force_calcs(V, rho0, phi, k, ax, Ay_i, beta_i, kf_i, kr_i, delta)

        # residues
        Rx_B = V_dot*ca.cos(beta_i) + forces['ax_b'] - (forces['Fx_b']/m)
        Ry_B = V_dot*ca.sin(beta_i) + forces['ay_b'] - (forces['Fy_b']/m)

        # add to performance metric J
        J += (Rx_B**2) + (Ry_B**2)

        Mz = a*(forces['fy_f']*ca.cos(delta) + forces['fx_f']*ca.sin(delta)) - b*forces['fy_r']

        # derivatives wrt ay
        d_ay = ay_range[i+1] - ay_range[i]
        if i < (N-2):
            opti.subject_to(beta[i+1] == beta[i] + d_ay*beta_prime[i])
            opti.subject_to(kappa_f[i+1] == kappa_f[i] + d_ay*kappa_f_prime[i])
            opti.subject_to(kappa_r[i+1] == kappa_r[i] + d_ay*kappa_r_prime[i])

        opti.subject_to(opti.bounded(-0.5, beta_i, 0.5))
        opti.subject_to(opti.bounded(-kappa_max, kf_i, 0))
        opti.subject_to(opti.bounded(-kappa_max, kr_i, kappa_max))



        # CONSTRAINTS
        # ignoring vertical force and pitching moment constraints bc we computed directly
        # TODO: yaw moment balance constraint?
        # opti.subject_to(ca.fabs(Mz) <= 0.01)

        # vehicle power constraint
        opti.subject_to((V*ca.fmax(forces['fx_r'], 0))/(P_max*1000) <= 1)

        # tire and slip angle constraints
        # TODO: do we need to move this outside the for loop?
        # opti.subject_to(opti.bounded(-alpha_max, forces['alpha_r'], alpha_max))
        # opti.subject_to(opti.bounded(-alpha_max, forces['alpha_f'], alpha_max))

        # opti.subject_to(opti.bounded(-kappa_max, kr_i, kappa_max))
        # opti.subject_to(opti.bounded(-kappa_max, kf_i, 0))

    # out of the for loop
    opti.minimize(J)
    
    # set initial guesses
    # beta_guess = np.arctan(b/(a+b)*np.tan(delta))
    opti.set_initial(beta, beta_guess)
    opti.set_initial(kappa_f, kf_guess)
    opti.set_initial(kappa_r, kr_guess)

    opti.set_initial(beta_prime, 0.0)
    opti.set_initial(kappa_f_prime, 0.0)
    opti.set_initial(kappa_r_prime, 0.0)

    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 2500}
    opti.solver('ipopt', opts)
    try:
        sol = opti.solve()
        beta_sol = sol.value(beta)
        kf_sol = sol.value(kappa_f)
        kr_sol = sol.value(kappa_r)

        mz_vals = []
        residues = []

        # calculating mz and residues and storing them
        for i in range(N):
            beta_i = beta_sol[i]
            kf_i = kf_sol[i]
            kr_i = kr_sol[i]
            Ay_i = ay_range[i]

            forces = sols_force_calcs(V, rho0, phi, k, ax, Ay_i, beta_i, kf_i, kr_i, delta)

            # residues
            Rx_B = V_dot*np.cos(beta_i) + forces['ax_b'] - (forces['Fx_b']/m)
            Ry_B = V_dot*np.sin(beta_i) + forces['ay_b'] - (forces['Fy_b']/m)
            # Rx_B = -forces['Fx_b']
            # Ry_B = m*Ay_i - forces['Fy_b']

            Mz = a*(forces['fy_f']*np.cos(delta) + forces['fx_f']*np.sin(delta)) - b*forces['fy_r']
            mz_vals.append(Mz)

            residue = np.sqrt((Rx_B**2) + (Ry_B**2))
            residues.append(residue)

            if i % 10 == 0 or abs(Ay_i/g) > 1.5:
                print(f"i = {i}, Ay={Ay_i/g:.3f}g, residue={residue:.2f}, mz={Mz:.1f}, beta={np.rad2deg(beta_i):.2f}, kf={kf_i:.4f}, kr={kr_i:.4f}")


        # mz_normalized = np.array(mz_vals)/(m*g*wheelbase)
        mz_normalized = np.array(mz_vals)
        ay_normalized = ay_range/g
        residue_threshold = 50
        residues = np.array(residues)

        return ay_normalized, mz_normalized, beta_sol, residues, kf_sol, kr_sol
            
    except:
        return None, None, None, None, None, None


def beta_ocp(beta, V, V_dot, ay_min, ay_max, delta_guess, kf_guess, kr_guess, N=50):
    global m, a, b, h, g, rho_air, P_max, A, Cd, Cs, Cl
    global rho0, phi, k
    global alpha_max, kappa_max
    
    ax = V_dot
    
    opti = ca.Opti()

    # states
    delta = opti.variable(N)
    kappa_f = opti.variable(N)
    kappa_r = opti.variable(N)

    # controls
    delta_prime = opti.variable(N-1)
    kappa_f_prime = opti.variable(N-1)
    kappa_r_prime = opti.variable(N-1)

    # Ay = opti.parameter(N)
    ay_range = np.linspace(ay_min, ay_max, N)

    J = 0
    for i in range(N-1):
        delta_i = delta[i]
        kf_i = kappa_f[i]
        kr_i = kappa_r[i]
        Ay_i = ay_range[i] # TODO: correct?
        Ay_i = ca.if_else(ca.fabs(Ay_i) > 0.1, Ay_i, ca.sign(Ay_i)*0.1)

        forces = force_calcs(V, rho0, phi, k, ax, Ay_i, beta, kf_i, kr_i, delta_i)

        # residues
        Rx_B = V_dot*ca.cos(beta) + forces['ax_b'] - (forces['Fx_b']/m)
        Ry_B = V_dot*ca.sin(beta) + forces['ay_b'] - (forces['Fy_b']/m)

        # add to performance metric J
        J += (Rx_B**2) + (Ry_B**2)

        Mz = a*(forces['fy_f']*ca.cos(delta_i) + forces['fx_f']*ca.sin(delta_i)) - b*forces['fy_r']

        # derivatives wrt ay
        d_ay = ay_range[i+1] - ay_range[i]
        if i < (N-2):
            opti.subject_to(delta[i+1] == delta[i] + d_ay*delta_prime[i])
            opti.subject_to(kappa_f[i+1] == kappa_f[i] + d_ay*kappa_f_prime[i])
            opti.subject_to(kappa_r[i+1] == kappa_r[i] + d_ay*kappa_r_prime[i])

        opti.subject_to(opti.bounded(-np.deg2rad(30), delta_i, np.deg2rad(30)))
        opti.subject_to(opti.bounded(-kappa_max, kf_i, 0))
        opti.subject_to(opti.bounded(-kappa_max, kr_i, kappa_max))



        # CONSTRAINTS
        # ignoring vertical force and pitching moment constraints bc we computed directly
        # TODO: yaw moment balance constraint?
        # opti.subject_to(ca.fabs(Mz) <= 0.01)

        # vehicle power constraint
        opti.subject_to((V*ca.fmax(forces['fx_r'], 0))/(P_max*1000) <= 1)

        # tire and slip angle constraints
        # TODO: do we need to move this outside the for loop?
        # opti.subject_to(opti.bounded(-alpha_max, forces['alpha_r'], alpha_max))
        # opti.subject_to(opti.bounded(-alpha_max, forces['alpha_f'], alpha_max))

        # opti.subject_to(opti.bounded(-kappa_max, kr_i, kappa_max))
        # opti.subject_to(opti.bounded(-kappa_max, kf_i, 0))

    # out of the for loop
    opti.minimize(J)
    
    # set initial guesses
    opti.set_initial(delta, delta_guess)
    opti.set_initial(kappa_f, kf_guess)
    opti.set_initial(kappa_r, kr_guess)

    opti.set_initial(delta_prime, 0.0)
    opti.set_initial(kappa_f_prime, 0.0)
    opti.set_initial(kappa_r_prime, 0.0)

    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 2500}
    opti.solver('ipopt', opts)
    try:
        sol = opti.solve()
        delta_sol = sol.value(delta)
        kf_sol = sol.value(kappa_f)
        kr_sol = sol.value(kappa_r)

        mz_vals = []
        residues = []

        # calculating mz and residues and storing them
        for i in range(N):
            delta_i = delta_sol[i]
            kf_i = kf_sol[i]
            kr_i = kr_sol[i]
            Ay_i = ay_range[i]

            forces = sols_force_calcs(V, rho0, phi, k, ax, Ay_i, beta, kf_i, kr_i, delta_i)

            # residues
            Rx_B = V_dot*np.cos(beta) + forces['ax_b'] - (forces['Fx_b']/m)
            Ry_B = V_dot*np.sin(beta) + forces['ay_b'] - (forces['Fy_b']/m)
            # Rx_B = -forces['Fx_b']
            # Ry_B = m*Ay_i - forces['Fy_b']

            Mz = a*(forces['fy_f']*np.cos(delta_i) + forces['fx_f']*np.sin(delta_i)) - b*forces['fy_r']
            mz_vals.append(Mz)

            residue = np.sqrt((Rx_B**2) + (Ry_B**2))
            residues.append(residue)

            if i % 10 == 0 or abs(Ay_i/g) > 1.5:
                print(f"i = {i}, Ay={Ay_i/g:.3f}g, residue={residue:.2f}, mz={Mz:.1f}, delta={np.rad2deg(delta_i):.2f}, kf={kf_i:.4f}, kr={kr_i:.4f}")


        # mz_normalized = np.array(mz_vals)/(m*g*wheelbase)
        mz_normalized = np.array(mz_vals)
        ay_normalized = ay_range/g
        residue_threshold = 50
        residues = np.array(residues)

        return ay_normalized, mz_normalized, delta_sol, residues, kf_sol, kr_sol
            
    except:
        return None, None, None, None, None, None

delta_min = -9
delta_max = 10 # will get us to 9
delta_step = 1
delta_range = np.arange(delta_min, delta_max, delta_step)

beta_max = np.deg2rad(15)
beta_n = 19
beta_range = np.linspace(-beta_max, beta_max, beta_n)

V = 15
V_dot = 0
rho0 = 0.5
phi = 0 # right helicoid
k = -1

curvature, torsion = curvature_torsion(rho0, phi, k)
a_centrip = (V**2)*curvature
ay_min = -2.5*g
ay_max = 2.5*g

results = {}

prev_beta = None
prev_kf = None
prev_kr = None

for delta in delta_range:
    print("delta = ", delta)
    delta_rad = np.deg2rad(delta)
    if prev_beta is not None:
        ay_norm, mz_norm, beta_sol, residues, kf_sol, kr_sol = ocp(delta_rad, V, V_dot, ay_min, ay_max, prev_beta, prev_kf, prev_kr)
    else:
        beta_guess = np.arctan(b/(a+b)*np.tan(delta_rad))
        ay_norm, mz_norm, beta_sol, residues, kf_sol, kr_sol = ocp(delta_rad, V, V_dot, ay_min, ay_max, beta_guess, 0.0, 0.0)


    if ay_norm  is not None:
        ay_norm = np.array(ay_norm)
        mz_norm = np.array(mz_norm)
        beta_sol = np.array(beta_sol)
        residues = np.array(residues)

        results[delta] = {
                'ay': ay_norm,
                'mz': mz_norm,
                'beta': beta_sol,
                'residues': residues
        
        }
        prev_beta = beta_sol
        prev_kf = kf_sol
        prev_kr = kr_sol
    else:
        print("{} failed".format(delta))

beta_results = {}
prev_delta = None
prev_kf = None
prev_kr = None
for beta in beta_range:
    print("beta = ", beta)
    if prev_delta is not None:
        ay_norm, mz_norm, delta_sol, residues, kf_sol, kr_sol = beta_ocp(beta, V, V_dot, ay_min, ay_max, prev_delta, prev_kf, prev_kr)
    else:
        delta_guess = 0
        ay_norm, mz_norm, delta_sol, residues, kf_sol, kr_sol = beta_ocp(beta, V, V_dot, ay_min, ay_max, delta_guess, 0.0, 0.0)

    if ay_norm is not None:
        ay_norm = np.array(ay_norm)
        mz_norm = np.array(mz_norm)
        delta_sol = np.array(delta_sol)
        residues = np.array(residues)

        beta_results[beta] = {
                'ay': ay_norm,
                'mz': mz_norm,
                'delta': delta_sol,
                'residues': residues
        
        }
        prev_delta = delta_sol
        prev_kf = kf_sol
        prev_kr = kr_sol
    else:
        print("{} failed".format(beta))



# plot
fig, ax = plt.subplots(figsize=(14,6))

for delta, data in results.items():
    delta_deg = np.round(np.rad2deg(delta), decimals=2)
    residues = data['residues']
    mask = residues <= 10
    if len(data['ay']) > 0:
        color = None
        if delta_deg > 0:
            color = 'red'
        elif delta_deg == 0:
            color = 'black'
        else:
            color = 'blue'

        ax.plot(data['ay'][mask], data['mz'][mask], '-', color=color, linewidth=1.5)

for beta, data in beta_results.items():
    beta = np.round(beta, decimals=2)
    residues = data['residues']
    mask = residues <= 5
    if len(data['ay']) > 0:
        color = None
        if beta > 0:
            color = 'red'
        elif beta == 0:
            color = 'black'
        else:
            color = 'blue'

        ax.plot(data['ay'][mask], data['mz'][mask], '-', color=color, linewidth=1.5)


ax.set_xlabel("Normalized lateral accel (g)", fontsize=12)
ax.set_ylabel("Yaw moment", fontsize=12)
ax.set_title(f"Yaw Moment Diagram for V={V} m/s")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

plt.tight_layout()
plt.show()

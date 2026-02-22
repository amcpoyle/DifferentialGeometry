import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def helicoid(rho, alpha, theta):
    x = rho*np.cos(alpha*theta)
    y = rho*np.sin(alpha*theta)
    z = theta

    return x, y, z

def cartesian_helicoid(alpha, rho_max, theta_max):
    rho_range = np.linspace(-rho_max, rho_max, 20)
    theta_range = np.linspace(-theta_max, theta_max, 20)
    R, T = np.meshgrid(rho_range, theta_range)
    
    X = R*np.cos(alpha*T)
    Y = R*np.sin(alpha*T)
    Z = T
    
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis')
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Helicoidal Ruled Surface")
    plt.show()

def new_helicoid(k, rho_max, theta_max):
    rho_range = np.linspace(-rho_max, rho_max, 20)
    theta_range = np.linspace(-theta_max, theta_max, 20)
    R, T = np.meshgrid(rho_range, theta_range)
    
    X = R*np.cos(T)
    Y = R*np.sin(T)
    Z = -k*T

    return X, Y, Z
    

def trajectory(k, rho_traj, theta):
    theta_traj = np.linspace(-theta, theta, 25)
    x_traj = rho_traj*np.cos(theta_traj)
    y_traj = rho_traj*np.sin(theta_traj)
    z_traj = -k*theta_traj

    return x_traj, y_traj, z_traj

alpha = 1
rho_max = 1
theta_max = np.pi
k = -1 

# cartesian_helicoid(alpha, rho_max, theta_max)
heli_x, heli_y, heli_z = new_helicoid(k, rho_max, theta_max)

fig = plt.figure(figsize=(14,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(heli_x, heli_y, heli_z, alpha=0.5, cmap='viridis')

rho_traj = rho_max/2
x_traj, y_traj, z_traj = trajectory(k, rho_traj, theta_max)

ax.plot(x_traj, y_traj, z_traj, color='red', linewidth=3, label='Trajectory')
ax.legend()

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Helicoidal Ruled Surface")
plt.tight_layout()
plt.show()

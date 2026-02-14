import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# surfaces of constant curvature

# surfaces of vanishing Gaussian curvature

# x = np.linspace(0, 1, 1000)
# y = np.linspace(0, 1, 1000)
# 
# X,Y = np.meshgrid(x, y)
# Z = np.cos(X) + np.sin(Y)
u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(0, 2*np.pi, 100)
U, V = np.meshgrid(u,v)

def torus(u, v, a, b):
    if b > a:
        return

    x = (a + b*np.cos(u))*np.cos(v)
    y = (a + b*np.cos(u))*np.sin(v)
    z = b*np.sin(u)

    return x, y, z

def graph_torus(a, b):
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, 2*np.pi, 100)
    U, V = np.meshgrid(u,v)

    x, y, z = torus(U, V, 1, 0.2)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0)
    # vertices = np.array(surf._vec[:3]).T
    ax.set_zlim(0, 1.01)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

# our open set U = S^1
r = 1
theta_range = np.linspace(0, 2*np.pi, 100)
x_values = np.zeros(len(theta_range))
y_values = np.zeros(len(theta_range))
# get all x, y values that satisfy this
for i, theta in enumerate(theta_range):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    x_values[i] = x
    y_values[i] = y

plt.plot(x_values, y_values)
plt.show()

# goal: compute the tangent plane of f at point p

# want to calculate the curvature at every point of the surface

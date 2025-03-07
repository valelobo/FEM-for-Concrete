# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 21:04:35 2025

@author: Leon Lobo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri

# =====================
# Pre-processing Stage
# =====================

# Beam geometry and material properties
L = 5.0          # Length of beam (m)
H = 1.0          # Height of beam (m)
E = 30e9         # Young's modulus (Pa)
nu = 0.2         # Poisson's ratio
thickness = 1.0  # Thickness (m)
force = -5000.0  # Applied force (N)

# Mesh parameters
nx = 15*10          # Number of elements along length
ny = 15*2           # Number of elements along height

# Generate nodes
x_coords = np.linspace(0, L, nx+1)
y_coords = np.linspace(0, H, ny+1)
nodes = np.array([(x, y) for x in x_coords for y in y_coords])
num_nodes = len(nodes)

# Generate elements (CST)
elements = []
for i in range(nx):
    for j in range(ny):
        # Lower triangle
        n1 = i*(ny+1) + j
        n2 = (i+1)*(ny+1) + j
        n3 = i*(ny+1) + j + 1
        elements.append([n1, n2, n3])
        
        # Upper triangle
        n1 = (i+1)*(ny+1) + j
        n2 = (i+1)*(ny+1) + j + 1
        n3 = i*(ny+1) + j + 1
        elements.append([n1, n2, n3])

elements = np.array(elements)

# ===================
# Processing Stage
# ===================

# Material matrix for plane stress
D = E/(1 - nu**2) * np.array([[1, nu, 0],
                             [nu, 1, 0],
                             [0, 0, (1 - nu)/2]])

# Initialize global stiffness matrix
dof = 2 * num_nodes
K = np.zeros((dof, dof))

# Assemble global stiffness matrix
for element in elements:
    n = element
    x = nodes[n, 0]
    y = nodes[n, 1]
    
    # Calculate element area
    A = 0.5 * np.abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
    
    # Calculate B matrix components
    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
    c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
    
    B = np.zeros((3, 6))
    for i in range(3):
        B[0, 2*i] = b[i]/(2*A)
        B[1, 2*i+1] = c[i]/(2*A)
        B[2, 2*i] = c[i]/(2*A)
        B[2, 2*i+1] = b[i]/(2*A)
    
    # Element stiffness matrix
    Ke = (thickness * A) * B.T @ D @ B
    
    # Assemble into global matrix
    indices = np.array([[2*n[0], 2*n[0]+1],
                       [2*n[1], 2*n[1]+1],
                       [2*n[2], 2*n[2]+1]]).flatten()
    
    for i, ii in enumerate(indices):
        for j, jj in enumerate(indices):
            K[ii, jj] += Ke[i, j]

# Apply boundary conditions (fixed left end)
fixed_nodes = [i for i in range(num_nodes) if nodes[i, 0] == 0]
fixed_dofs = []
for n in fixed_nodes:
    fixed_dofs.extend([2*n, 2*n+1])

# Apply force (right end top node)
force_node = [i for i, (x, y) in enumerate(nodes) 
             if np.isclose(x, L) and np.isclose(y, H)][0]
F = np.zeros(dof)
F[2*force_node + 1] = force

# Modify stiffness matrix and force vector for BCs
K_red = np.delete(np.delete(K, fixed_dofs, axis=0), fixed_dofs, axis=1)
F_red = np.delete(F, fixed_dofs)

# Solve system
U_red = np.linalg.solve(K_red, F_red)

# Reconstruct full displacement vector
U = np.zeros(dof)
free_dofs = np.delete(np.arange(dof), fixed_dofs)
U[free_dofs] = U_red

# ======================
# Post-processing Stage
# ======================

# Create triangulation object
triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)

# Initialize stress storage arrays
sigma_xx = []  # X-direction normal stress
sigma_yy = []  # Y-direction normal stress
tau_xy = []    # Shear stress

for element in elements:
    n = element
    x = nodes[n, 0]
    y = nodes[n, 1]
    
    # Element area calculation
    A = 0.5 * np.abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
    
    # B matrix components
    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
    c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
    
    B = np.zeros((3, 6))
    for i in range(3):
        B[0, 2*i] = b[i]/(2*A)
        B[1, 2*i+1] = c[i]/(2*A)
        B[2, 2*i] = c[i]/(2*A)
        B[2, 2*i+1] = b[i]/(2*A)
    
    # Element displacements
    U_e = U[np.array([2*n[0], 2*n[0]+1, 
             2*n[1], 2*n[1]+1,
             2*n[2], 2*n[2]+1])]
    
    # Calculate all stress components
    stress = D @ B @ U_e
    sigma_xx.append(stress[0])  # σ_xx component
    sigma_yy.append(stress[1])  # σ_yy component
    tau_xy.append(stress[2])    # τ_xy component

# Nodal stress averaging
nodal_sigma_xx = np.zeros(num_nodes)
nodal_sigma_yy = np.zeros(num_nodes)
nodal_tau_xy = np.zeros(num_nodes)
element_count = np.zeros(num_nodes)

for idx, element in enumerate(elements):
    for node in element:
        nodal_sigma_xx[node] += sigma_xx[idx]
        nodal_sigma_yy[node] += sigma_yy[idx]
        nodal_tau_xy[node] += tau_xy[idx]
        element_count[node] += 1

nodal_sigma_xx /= element_count
nodal_sigma_yy /= element_count
nodal_tau_xy /= element_count

# Deformation visualization
scale_factor = 5e3  # Displacement scaling for visualization
deformed_nodes = nodes + U.reshape(-1, 2) * scale_factor

# Deformation visualization with triangulation
plt.figure(figsize=(12, 4))
plt.title("Deformed vs Undeformed Shape")
plt.gca().set_aspect('equal')

# Plot undeformed mesh
plt.triplot(triangulation, color='grey', linewidth=0.5)

# Create deformed triangulation
deformed_triangulation = tri.Triangulation(deformed_nodes[:, 0], 
                                          deformed_nodes[:, 1], elements)

# Plot deformed mesh
plt.triplot(deformed_triangulation, color='red', linewidth=0.5)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.show()

# X-direction Normal Stress Plot
plt.figure(figsize=(8, 4))
plt.title(r"Longitudinal Normal Stress $\sigma_{xx}$")
tcf = plt.tricontourf(triangulation, nodal_sigma_xx, levels=20, cmap='jet')
plt.colorbar(tcf, label='Stress (Pa)')
plt.gca().set_aspect('equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.show()

# Y-direction Normal Stress Plot
plt.figure(figsize=(8, 4))
plt.title(r"Transverse Normal Stress $\sigma_{yy}$")
tcf = plt.tricontourf(triangulation, nodal_sigma_yy, levels=20, cmap='jet')
plt.colorbar(tcf, label='Stress (Pa)')
plt.gca().set_aspect('equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.show()

# Shear Stress Plot
plt.figure(figsize=(8, 4))
plt.title(r"Shear Stress $\tau_{xy}$")
tcf = plt.tricontourf(triangulation, nodal_tau_xy, levels=20, cmap='jet')
plt.colorbar(tcf, label='Stress (Pa)')
plt.gca().set_aspect('equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.show()
